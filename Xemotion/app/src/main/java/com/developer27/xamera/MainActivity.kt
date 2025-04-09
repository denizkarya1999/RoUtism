package com.developer27.xemotion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.preference.PreferenceManager
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding
import com.developer27.xemotion.inference.EmotionActivity
import com.developer27.xemotion.videoprocessing.ProcessedFrameRecorder
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Timer
import java.util.TimerTask

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var tfliteInterpreter: Interpreter? = null
    private var processedFrameRecorder: ProcessedFrameRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    // Flag for tracking (start/stop tracking mode)
    private var isRecording = false
    // Flag for frame processing
    private var isProcessing = false
    private var isProcessingFrame = false

    // Variable for inference result.
    private var inferenceResult = ""

    // Stores the tracking coordinates.
    private var trackingCoordinates: String = ""

    // New flag to clear prediction when returning from an external intent.
    private var shouldClearPrediction = false

    // Timer for periodic export every second
    private var exportTimer: Timer? = null
    // Counter for number of batches exported.
    private var batchCount = 0

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Prevent screen from turning off
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Lock screen orientation to portrait
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        // Install the splash screen (Android 12+)
        installSplashScreen()

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Hide the processed frame view initially.
        viewBinding.processedFrameView.visibility = View.GONE
        // Set default text for the predicted emotion TextView.
        viewBinding.predictedEmotionTextView.text = "No Prediction Yet"

        // When the title container is clicked, open the URL in a browser.
        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.developer27.com"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                val camGranted = permissions[Manifest.permission.CAMERA] ?: false
                val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
                if (camGranted && micGranted) {
                    if (viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    } else {
                        viewBinding.viewFinder.surfaceTextureListener = textureListener
                    }
                } else {
                    Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
                }
            }

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Set up the Start Tracking button.
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                // Stop tracking and update prediction.
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        // Set up the Switch Camera, About, and Settings buttons.
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Set up the Clear Prediction button ("C") under the zoom buttons.
        viewBinding.clearPredictionButton.setOnClickListener {
            // Stop tracking if active.
            if (isRecording) {
                stopProcessingAndRecording()
            }
            // Reset the prediction panel.
            viewBinding.predictedEmotionTextView.text = "No Prediction Yet"
        }

        // Register the inference button (added under the "C" button)
        val inferenceButton: Button = findViewById(R.id.inferenceButton)
        inferenceButton.setOnClickListener {
            launchEmotionActivity()
        }

        // Load ML models.
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }
    }

    /**
     * Launches the EmotionActivity which allows the user to select an image for emotion inference.
     */
    private fun launchEmotionActivity() {
        val intent = Intent(this, EmotionActivity::class.java)
        startActivity(intent)
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // Clear any previous trace data and reset the batch counter.
        videoProcessor?.reset()
        batchCount = 0

        // Start a timer that exports the drawn line every 0.5 seconds and then clears it.
        exportTimer = Timer()
        exportTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                runOnUiThread {
                    exportCurrentLine()
                    videoProcessor?.reset()
                }
            }
        }, 500, 500) // 0.5 second delay and repeat every 0.5 seconds
    }

    // Helper function to export the current drawn line as a JPEG image.
    private fun exportCurrentLine() {
        try {
            val traceBitmap = videoProcessor?.exportTraceForDataCollection()
            if (traceBitmap != null) {
                val screenshotPath = getProcessedImageOutputPath()
                val screenshotFile = File(screenshotPath)
                FileOutputStream(screenshotFile).use { outputStream ->
                    traceBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                    outputStream.flush()
                }
                // Increment the batch counter (no individual toast shown)
                batchCount++
                Log.d("MainActivity", "Batch #$batchCount exported: $screenshotPath")
            } else {
                Log.e("MainActivity", "Failed to export trace: Bitmap is null.")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error exporting trace", e)
        }
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false

        // Cancel the periodic export timer.
        exportTimer?.cancel()
        exportTimer = null

        // Optionally perform a final export.
        try {
            val traceBitmap = videoProcessor?.exportTraceForDataCollection()
            if (traceBitmap != null) {
                val screenshotPath = getProcessedImageOutputPath()
                val screenshotFile = File(screenshotPath)
                FileOutputStream(screenshotFile).use { outputStream ->
                    traceBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                    outputStream.flush()
                }
                // Increment the batch counter for the final export.
                batchCount++
                Log.d("MainActivity", "Final batch exported: $screenshotPath")
            } else {
                Log.e("MainActivity", "Failed to export trace: Bitmap is null.")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error exporting trace", e)
        }

        // Update UI and clean up the overlay.
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        val outputPath = getProcessedImageOutputPath()
        processedFrameRecorder = ProcessedFrameRecorder(outputPath)
        with(Settings.ExportData) {
            if (frameIMG) {
                val bitmap = videoProcessor?.exportTraceForDataCollection()
                if (bitmap != null) {
                    // processedFrameRecorder?.save(bitmap)
                }
            }
        }

        inferenceResult = "Study ML Inference in Rollity" // Temporary message
        viewBinding.predictedEmotionTextView.text = inferenceResult

        // Retrieve tracking coordinates.
        trackingCoordinates = videoProcessor?.getTrackingCoordinatesString() ?: ""

        // Show a single Toast message with the total number of batches saved.
        Toast.makeText(this, "$batchCount batches have been saved", Toast.LENGTH_LONG).show()
    }

    // Helper function to crop a bitmap to its non-white content.
    private fun cropToNonWhite(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                // Check if pixel is not white.
                if (bitmap.getPixel(x, y) != android.graphics.Color.WHITE) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        // If no non-white pixels found, return a 1x1 white image.
        if (minX > maxX || minY > maxY) {
            return Bitmap.createBitmap(1, 1, bitmap.config).apply { eraseColor(android.graphics.Color.WHITE) }
        }
        return Bitmap.createBitmap(bitmap, minX, minY, maxX - minX + 1, maxY - minY + 1)
    }

    // Helper function to capture a view (including its drawing/overlay) as a Bitmap.
    private fun captureView(view: View): Bitmap {
        val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        view.draw(canvas)
        return bitmap
    }

    // TODO: Study how digit inference model is implemented in Rollity.
    private fun runDigitRecognitionInference(): String {
        val digitBitmap = videoProcessor?.exportTraceForInference()
        if (digitBitmap == null) {
            Log.e("MainActivity", "No digit image available for inference")
            return "Error"
        }
        val grayBitmap = convertToGrayscale(digitBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        val outputArray = Array(1) { FloatArray(10) }
        if (tfliteInterpreter == null) {
            Log.e("MainActivity", "Digit model interpreter not set")
            return "Error"
        }
        tfliteInterpreter?.run(inputBuffer, outputArray)
        val predictedDigit = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        Log.d("MainActivity", "Digit model predicted: $predictedDigit")
        return predictedDigit.toString()
    }

    // TODO: Study how digit inference model is implemented in Rollity.
    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    // TODO: Study how digit inference model is implemented in Rollity.
    private fun convertBitmapToGrayscaleByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputSize = bitmap.width * bitmap.height
        val byteBuffer = ByteBuffer.allocateDirect(inputSize * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF).toFloat()
            val normalized = r / 255.0f
            byteBuffer.putFloat(normalized)
        }
        return byteBuffer
    }

    // TODO: Study how parameters are passing to the ARCore and think about how this will happen in RoUtism.
    private fun launch3DActivity() {
        // Send the accumulated prediction information to the 3D launcher.
        val intent = Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java)
        intent.putExtra("LETTER_KEY", viewBinding.predictedEmotionTextView.text.toString())
        val pathCoordinates = if (trackingCoordinates.isNotEmpty()) {
            trackingCoordinates
        } else {
            "0.0,0.0,0.0;5.0,10.0,-5.0;-5.0,15.0,10.0;20.0,-5.0,5.0;-10.0,0.0,-10.0;10.0,-15.0,15.0;0.0,20.0,-5.0"
        }
        intent.putExtra("PATH_COORDINATES", pathCoordinates)
        shouldClearPrediction = true
        startActivity(intent)
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, preprocessedBitmap) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                        // Removed ProcessedVideoRecorder parts.
                    }
                }
                isProcessingFrame = false
            }
        }
    }

    private fun getProcessedImageOutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val routismDir = File(picturesDir, "RoEmotion_ML_Training_Data")
        if (!routismDir.exists()) {
            routismDir.mkdirs()
        }
        return File(routismDir, "Inference_${System.currentTimeMillis()}.jpg").absolutePath
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(modelName)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val options = Interpreter.Options().apply {
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        var delegateAdded = false
                        try {
                            val nnApiDelegate = NnApiDelegate()
                            options.addDelegate(nnApiDelegate)
                            delegateAdded = true
                            Log.d("MainActivity", "NNAPI delegate added successfully.")
                        } catch (e: Exception) {
                            Log.d("MainActivity", "NNAPI delegate unavailable, falling back to GPU delegate.", e)
                        }
                        if (!delegateAdded) {
                            try {
                                val gpuDelegate = GpuDelegate()
                                options.addDelegate(gpuDelegate)
                                Log.d("MainActivity", "GPU delegate added successfully.")
                            } catch (e: Exception) {
                                Log.d("MainActivity", "GPU delegate unavailable, will use CPU only.", e)
                            }
                        }
                        when (modelName) {
                            "YOLOv3_float32.tflite" -> {
                                videoProcessor?.setInterpreter(Interpreter(loadMappedFile(bestLoadedPath), options))
                            }
                            else -> Log.d("MainActivity", "No model processing method defined for $modelName")
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.d("MainActivity", "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile.absolutePath
            }
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                    }
                    output.flush()
                }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
            ""
        }
    }

    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
        // When returning from an external intent, clear the prediction.
        if (shouldClearPrediction) {
            viewBinding.predictedEmotionTextView.text = "No Prediction Yet"
            shouldClearPrediction = false
        }
    }

    override fun onPause() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}