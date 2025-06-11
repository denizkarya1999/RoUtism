package com.developer27.xemotion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.CountDownTimer
import android.os.Environment
import android.preference.PreferenceManager
import android.text.InputType
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.core.view.isVisible
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding
import com.developer27.xemotion.inference.PyTorchClassifier
import com.developer27.xemotion.inference.PyTorchModuleLoader
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
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

    // Private global variables
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var yoloInterpreter: Interpreter? = null
    private var videoProcessor: VideoProcessor? = null
    private var arTimer: CountDownTimer? = null

    // PyTorch model for classification
    private lateinit var emotionClassifier: PyTorchClassifier

    // Tracking flags
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    // Timer for periodic export
    private var exportTimer: Timer? = null
    private var batchCount = 0

    // For clearing predictions when returning from an external intent
    private var shouldClearPrediction = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        var isArMode: Boolean = false
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

        // Load PyTorch model directly to confirm
        val module = PyTorchModuleLoader.loadModule(this, "resnet50_emotion.pt")
        Log.d(TAG, "PyTorch model loaded successfully: $module")

        // Mode toggle button. If user sees "AR Mode," that means we are currently in CV mode,
        // and they can tap to switch to AR mode.
        viewBinding.modeToggleButton.setOnClickListener {
            isArMode = (viewBinding.modeToggleButton.text == "AR Mode")
            viewBinding.modeToggleButton.text = if (isArMode) "CV Mode" else "AR Mode"
            viewBinding.modeToggleButton.backgroundTintList =
                ColorStateList.valueOf(
                    ContextCompat.getColor(this,
                        if (isArMode) R.color.red else R.color.green
                    )
                )
            updateUiForMode(isArMode)
        }

        // Apply initial UI state (we start in CV mode by default)
        updateUiForMode(false)

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
            startActivity(Intent(this, AboutXemotionActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        viewBinding.clearPredictionButton.setOnClickListener {
            // If recording was ongoing, stop it
            if (isRecording) stopProcessingAndRecording()
            // Clear out the classification label from the VideoProcessor
            videoProcessor?.classificationLabel = ""
        }

        // Load YOLO model (for object detection) on a background thread
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")

        // in MainActivity.kt, wherever you initialize:
        emotionClassifier = PyTorchClassifier.fromAsset(this, "resnet50_emotion.pt")

        // Set up pinch-to-zoom or other camera zoom controls
        cameraHelper.setupZoomControls()
    }

    /**
     * Enable/disable controls based on AR/CV mode.
     */
    private fun updateUiForMode(isArModeFunc: Boolean) {
        if (isArMode) {
            // 1) Prompt for AR session duration
            val marginPx = (16 * resources.displayMetrics.density).toInt()
            val input = EditText(this).apply {
                hint = "Minutes"
                inputType = InputType.TYPE_CLASS_NUMBER
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    setMargins(marginPx, marginPx, marginPx, 0)
                }
            }

            AlertDialog.Builder(this)
                .setTitle("AR Session Duration")
                .setMessage("Enter how many minutes to run AR mode:")
                .setView(input)
                .setCancelable(false)
                .setPositiveButton("Start") { _, _ ->
                    //First stop processing to clean the background processes
                    stopProcessingAndRecording()

                    val minutes = input.text.toString().toLongOrNull() ?: 1L
                    val duration = minutes.coerceAtLeast(1) * 60_000L

                    // 2) Switch UI into AR mode
                    with(viewBinding) {
                        startProcessingButton.isVisible    = false
                        modeToggleButton.isVisible         = false
                        titleContainer.isVisible           = false
                        switchCameraButton.isVisible       = false
                        settingsButton.isVisible           = false
                        aboutButton.isVisible              = false
                        clearPredictionButton.isVisible    = false
                        zoomInButton.isVisible             = false
                        zoomOutButton.isVisible            = false
                    }

                    // Force AR rolling shutter in camera helper
                    cameraHelper.forceArRollingShutter()

                    // 3) Begin processing & recording in AR mode
                    startProcessingAndRecording()

                    // 4) Schedule automatic exit from AR mode
                    arTimer?.cancel()
                    arTimer = object : CountDownTimer(duration, 1_000L) {
                        override fun onTick(millisUntilFinished: Long) {
                            // (optional) update a UI element with remaining time
                        }
                        override fun onFinish() {
                            // AR session time up -> revert to normal mode
                            isArMode = false
                            viewBinding.modeToggleButton.text = "AR Mode"
                            viewBinding.modeToggleButton.backgroundTintList =
                                ColorStateList.valueOf(
                                    ContextCompat.getColor(this@MainActivity, R.color.green)
                                )
                            updateUiForMode(false)
                        }
                    }.start()
                }
                .setNegativeButton("Cancel") { _, _ ->
                    // If user cancels AR start, explicitly set isArMode = false:
                    isArMode = false
                    // Revert the UI to CV mode
                    viewBinding.modeToggleButton.text = "AR Mode"
                    viewBinding.modeToggleButton.backgroundTintList =
                        ColorStateList.valueOf(ContextCompat.getColor(this, R.color.green))
                    updateUiForMode(false)
                }
                .show()

        } else {
            // Cancel any in-flight timer
            arTimer?.cancel()

            // Restore normal UI for CV mode
            with(viewBinding) {
                modeToggleButton.text = "AR Mode"
                modeToggleButton.backgroundTintList =
                    ContextCompat.getColorStateList(this@MainActivity, R.color.green)
                startProcessingButton.isVisible    = true
                modeToggleButton.isVisible         = true
                titleContainer.isVisible           = true
                switchCameraButton.isVisible       = true
                settingsButton.isVisible           = true
                aboutButton.isVisible              = true
                clearPredictionButton.isVisible    = true
                zoomInButton.isVisible             = true
                zoomOutButton.isVisible            = true
            }

            // Stop processing & recording
            stopProcessingAndRecording()
        }
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

        // Use 500 ms interval if CONTOUR or default; 700 ms if YOLO
        val intervalMs = when (Settings.DetectionMode.current) {
            Settings.DetectionMode.Mode.CONTOUR -> 500L
            Settings.DetectionMode.Mode.YOLO    -> 700L
            else -> 500L // fallback
        }

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
        }, intervalMs, intervalMs)
    }

    private fun stopProcessingAndRecording() {
        // If we’re not recording, no need to stop
        if (!isRecording) return

        isRecording = false
        isProcessing = false

        exportTimer?.cancel()
        exportTimer = null

        // Final export of leftover frames
        try {
            val traceBitmap = videoProcessor?.exportTraceForDataCollection()
            if (traceBitmap != null) {
                saveBatchAndRunInference(traceBitmap)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting trace", e)
        }

        if (Settings.ExportData.frameIMG) {
            Toast.makeText(this, "$batchCount batches have been saved", Toast.LENGTH_LONG).show()
        }

        // Reset UI text & visibility
        viewBinding.startProcessingButton.text = getString(R.string.start_capture)
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        videoProcessor?.classificationLabel = ""
    }

    /**
     * Save the trace bitmap to disk (as line<N>.jpg) and run PyTorch-based classification on it.
     */
    private fun saveBatchAndRunInference(traceBitmap: Bitmap) {
        // 1) Save file, only if user enabled saving
        if (Settings.ExportData.frameIMG) {
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
        }

        // 2) Classify using PyTorch
        val (bestLabel, probs) = emotionClassifier.classifyLine(traceBitmap)

        // Derive confidence
        val bestIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = probs[bestIndex] * 100f
        val textResult = "$bestLabel (${String.format("%.1f", confidence)}%)"

        // 3) Put label inside the bounding box next time we detect something
        videoProcessor?.classificationLabel = textResult

        // 4) (Optional) Log it to a file
        appendPredictionToLog(textResult)
    }

    /**
     * Write the predictions to a text log file in Documents,
     * but only if the user enabled prediction-logging via Settings.
     */
    private fun appendPredictionToLog(prediction: String) {
        if (!Settings.ExportData.enablePredictionLogging) {
            // User has turned prediction logging off → do nothing.
            Log.d(TAG, "Prediction logging disabled; skipping append.")
            return
        }

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
     * Generate a file path for saving the processed image as "line<N>.jpg",
     * where N is batchCount+1.
     */
    private fun getProcessedImageOutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val roEmotionDir = File(picturesDir, "Exported Lines from Xemotion")
        if (!roEmotionDir.exists()) {
            roEmotionDir.mkdirs()
        }
        // Use batchCount+1 to name files sequentially: line1.jpg, line2.jpg, ...
        val fileName = "Line (${batchCount + 1}).jpg"
        return File(roEmotionDir, fileName).absolutePath
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
