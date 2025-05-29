package com.developer27.xemotion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.preference.PreferenceManager
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding
import com.developer27.xemotion.inference.PyTorchClassifier
import com.developer27.xemotion.inference.PyTorchModuleLoader
import com.developer27.xemotion.secondaryprocessing.TemplateMatcher
import com.developer27.xemotion.videoprocessing.ProcessedFrameRecorder
import com.developer27.xemotion.videoprocessing.VideoProcessor
import com.developer27.xemotion.videoprocessing.YOLOHelper
import com.google.android.filament.utils.Utils
import org.opencv.core.Mat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.Timer
import java.util.TimerTask

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var yoloInterpreter: Interpreter? = null
    private var processedFrameRecorder: ProcessedFrameRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    // PyTorch model for classification
    private lateinit var emotionClassifier: PyTorchClassifier

    // Tracking flags
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    // Timer for periodic export
    private var exportTimer: Timer? = null
    private var batchCount = 0

    // Template Matcher and reference templates (train/val)
    private lateinit var matcher: TemplateMatcher
    private lateinit var trainTemplates: Map<String, List<Mat>>
    private lateinit var valTemplates:   Map<String, List<Mat>>

    // For clearing predictions when returning from an external intent
    private var shouldClearPrediction = false

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
        private const val TAG = "MainActivity"
    }

    /**
     * Our TextureView listener for camera preview.
     */
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            Log.d(TAG, "onSurfaceTextureAvailable: $width x $height")
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
        // Prevent screen from dimming
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Lock screen orientation to portrait
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        // Android 12+ splash screen
        installSplashScreen()

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Hide processed frame initially
        viewBinding.processedFrameView.visibility = View.GONE

        // Set the texture listener
        viewBinding.viewFinder.surfaceTextureListener = textureListener

        // Title container -> open your website
        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.developer27.com"
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        }

        // Request permissions
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
                val camGranted = perms[Manifest.permission.CAMERA] ?: false
                val micGranted = perms[Manifest.permission.RECORD_AUDIO] ?: false
                if (camGranted && micGranted) {
                    if (viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    }
                } else {
                    Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
                }
            }
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Instantiate TemplateMatcher to eagerly load references
        matcher = TemplateMatcher(assets)

        // Load templates from assets/dataset
        trainTemplates = matcher.loadTemplates("dataset/train")
        valTemplates   = matcher.loadTemplates("dataset/val")

        // Load PyTorch model directly to confirm
        val module = PyTorchModuleLoader.loadModule(this, "resnet50_emotion.pt")
        Log.d(TAG, "PyTorch model loaded successfully: $module")

        // Set button actions
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        // (Optional) Clear prediction button is no longer needed for text,
        // but you can still hide or remove it in your layout if desired.
        viewBinding.clearPredictionButton.setOnClickListener {
            // If recording was ongoing, stop it
            if (isRecording) stopProcessingAndRecording()
            // Clear out the classification label from the VideoProcessor
            videoProcessor?.classificationLabel = ""
        }

        // Load YOLO model (for object detection) on a background thread
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")

        // Create PyTorchClassifier (for emotion classification)
        emotionClassifier = PyTorchClassifier(this, "resnet50_emotion.pt")

        // Set up pinch-to-zoom or other camera zoom controls
        cameraHelper.setupZoomControls()
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true

        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        videoProcessor?.reset()
        batchCount = 0

        // Periodically export a "trace" bitmap and classify it
        exportTimer = Timer()
        exportTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                runOnUiThread {
                    val traceBitmap = videoProcessor?.exportTraceForDataCollection()
                    if (traceBitmap != null) {
                        saveBatchAndRunInference(traceBitmap)
                    }
                    videoProcessor?.reset()
                }
            }
        }, 500, 500) // adjust times as needed
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false

        exportTimer?.cancel()
        exportTimer = null

        // Attempt one final export if available
        try {
            val traceBitmap = videoProcessor?.exportTraceForDataCollection()
            if (traceBitmap != null) {
                saveBatchAndRunInference(traceBitmap)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting trace", e)
        }

        Toast.makeText(this, "$batchCount batches have been saved", Toast.LENGTH_LONG).show()

        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        // Clear classification label from the bounding-box overlay
        videoProcessor?.classificationLabel = ""
    }

    /**
     * Save the trace bitmap to disk and run PyTorch-based classification on it.
     */
    private fun saveBatchAndRunInference(traceBitmap: Bitmap) {
        // 1) Save file
        val screenshotPath = getProcessedImageOutputPath()
        val screenshotFile = File(screenshotPath)
        try {
            FileOutputStream(screenshotFile).use { outputStream ->
                traceBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                outputStream.flush()
            }
            batchCount++
            Log.d(TAG, "Batch #$batchCount exported: $screenshotPath")
        } catch (e: IOException) {
            Log.e(TAG, "Error saving batch: ${e.message}")
        }

        // 2) Classify using PyTorch
        val (bestLabel, probs) = emotionClassifier.classifyLine(traceBitmap)

        // Display or log results if you want more detail
        val bestIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = probs[bestIndex] * 100f
        val textResult = "$bestLabel (${String.format("%.1f", confidence)}%)"

        // 3) Instead of showing in a TextView, store it in VideoProcessor
        //    so the label is drawn inside the bounding box next time we detect something.
        videoProcessor?.classificationLabel = textResult

        // 4) Also log the prediction to a file (optional)
        appendPredictionToLog(textResult)
    }

    /**
     * Write the predictions to a text log file in Documents.
     */
    private fun appendPredictionToLog(prediction: String) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val line = "$timestamp => $prediction"

        @Suppress("DEPRECATION")
        val docDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (!docDir.exists()) {
            docDir.mkdirs()
        }
        val logFile = File(docDir, "RoEmotion_Predictions_Log.txt")

        try {
            logFile.appendText("$line\n")
        } catch (e: IOException) {
            Log.e(TAG, "Error writing prediction log: ${e.message}")
        }
    }

    /**
     * Process frame with VideoProcessor for overlays/detections, then display.
     */
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, _) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    }
                }
                isProcessingFrame = false
            }
        }
    }

    /**
     * Generate a file path for saving the processed image.
     */
    private fun getProcessedImageOutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val roEmotionDir = File(picturesDir, "RoEmotion_ML_Training_Data")
        if (!roEmotionDir.exists()) {
            roEmotionDir.mkdirs()
        }
        return File(roEmotionDir, "Inference_${System.currentTimeMillis()}.jpg").absolutePath
    }

    /**
     * Load the YOLO TFLite model on a background thread; sets up the Interpreter for detection.
     */
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
                            Log.d(TAG, "NNAPI delegate added.")
                        } catch (e: Exception) {
                            Log.d(TAG, "NNAPI delegate unavailable, fallback to GPU", e)
                        }
                        if (!delegateAdded) {
                            try {
                                val gpuDelegate = GpuDelegate()
                                options.addDelegate(gpuDelegate)
                                Log.d(TAG, "GPU delegate added.")
                            } catch (e: Exception) {
                                Log.d(TAG, "GPU delegate unavailable, CPU only.", e)
                            }
                        }
                        if (modelName == "YOLOv3_float32.tflite") {
                            yoloInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                            // Pass this interpreter to VideoProcessor for object detection
                            videoProcessor?.setInterpreter(yoloInterpreter!!)
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.e(TAG, "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    /**
     * Map a TFLite file into memory.
     */
    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    /**
     * Copy a model file from assets to internal storage, returning its path.
     */
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
            Log.e(TAG, "Error copying asset $assetName: ${e.message}")
            ""
        }
    }

    /**
     * Toggle between front/rear cameras. Closes/reopens camera session.
     */
    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) stopProcessingAndRecording()
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()

        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) {
            cameraHelper.openCamera()
        }

        if (shouldClearPrediction) {
            videoProcessor?.classificationLabel = ""
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

    /**
     * Check if all required permissions are granted.
     */
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}