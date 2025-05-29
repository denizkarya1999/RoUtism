package com.developer27.xemotion.camera

import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Environment
import android.util.Log
import android.util.Size
import android.view.Surface
import android.widget.Toast
import com.developer27.xemotion.MainActivity
import com.developer27.xemotion.databinding.ActivityMainBinding
import java.io.File
import kotlin.math.max

/**
 * TempRecorderHelper handles raw video recording via MediaRecorder
 * while real-time processing is done in VideoProcessor.
 */
class TempRecorderHelper(
    private val mainActivity: MainActivity,
    private val cameraHelper: CameraHelper,
    private val sharedPreferences: android.content.SharedPreferences,
    private val viewBinding: ActivityMainBinding
) {
    private var mediaRecorder: MediaRecorder? = null
    private var outputFile: File? = null

    /**
     * Start recording raw video if the camera is ready.
     */
    fun startRecordingVideo() {
        if (cameraHelper.cameraDevice == null) {
            Toast.makeText(
                mainActivity,
                "CameraDevice not ready.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        // Clean up old recorder if still open
        mediaRecorder?.apply {
            try { stop() } catch (_: Exception) {}
            reset()
            release()
        }
        mediaRecorder = null

        try {
            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            }

            // Detect phone orientation to correct final video orientation
            setRecorderOrientation()

            // Create public Movies directory if needed
            @Suppress("DEPRECATION")
            val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
            if (moviesDir == null || (!moviesDir.exists() && !moviesDir.mkdirs())) {
                Toast.makeText(
                    mainActivity,
                    "Cannot access public Movies folder.",
                    Toast.LENGTH_LONG
                ).show()
                return
            }

            // e.g., "Xamera_1673783432219.mp4"
            outputFile = File(moviesDir, "Xamera_${System.currentTimeMillis()}.mp4")
            mediaRecorder?.setOutputFile(outputFile!!.absolutePath)

            // Example size: 1280x720
            val recordSize = cameraHelper.videoSize ?: Size(1280, 720)
            mediaRecorder?.apply {
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setVideoEncodingBitRate(2_000_000)
                setVideoFrameRate(30)
                setVideoSize(recordSize.width, recordSize.height)
                prepare()
            }

            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            // Re-use the preview size for the texture
            cameraHelper.previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val previewSurface = Surface(texture)
            val recorderSurface = mediaRecorder!!.surface

            // Build the capture request for RECORD
            cameraHelper.captureRequestBuilder =
                cameraHelper.cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)

            // Add preview + recorder surfaces
            cameraHelper.captureRequestBuilder?.addTarget(previewSurface)
            cameraHelper.captureRequestBuilder?.addTarget(recorderSurface)

            // Apply auto/manual exposure logic
            applyRollingShutterForRecording(cameraHelper.captureRequestBuilder)

            // Now create the capture session
            cameraHelper.cameraDevice?.createCaptureSession(
                listOf(previewSurface, recorderSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraHelper.cameraCaptureSession = session
                        try {
                            cameraHelper.cameraCaptureSession?.setRepeatingRequest(
                                cameraHelper.captureRequestBuilder!!.build(),
                                null,
                                cameraHelper.backgroundHandler
                            )
                            mediaRecorder?.start()
                        } catch (e: CameraAccessException) {
                            Toast.makeText(
                                mainActivity,
                                "Failed to start recording: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                            e.printStackTrace()
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(
                            mainActivity,
                            "Capture session config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                cameraHelper.backgroundHandler
            )

        } catch (e: Exception) {
            Log.e("TempRecorderHelper", "MediaRecorder error: ${e.message}", e)
            Toast.makeText(
                mainActivity,
                "Cannot record: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            mediaRecorder?.reset()
            mediaRecorder?.release()
            mediaRecorder = null
        }
    }

    /**
     * Stop recording and finalize the file
     */
    fun stopRecordingVideo() {
        if (mediaRecorder == null) return
        try {
            mediaRecorder?.stop()
        } catch (e: Exception) {
            Log.e("TempRecorderHelper", "Error stopping recording: ${e.message}", e)
        }
        mediaRecorder?.reset()
        mediaRecorder?.release()
        mediaRecorder = null

        // Restore normal camera preview
        cameraHelper.createCameraPreview()

        // Show the path of saved file if it exists
        outputFile?.let { file ->
            if (file.exists()) {
                Toast.makeText(
                    mainActivity,
                    "Raw video saved:\n${file.absolutePath}",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                Toast.makeText(mainActivity, "No output file found.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * FULL AUTO or FULL MANUAL for recording, just like in CameraHelper.
     */
    private fun applyRollingShutterForRecording(builder: CaptureRequest.Builder?) {
        if (builder == null) return

        val cameraId = cameraHelper.getCameraId()
        val characteristics = cameraHelper.cameraManager.getCameraCharacteristics(cameraId)

        // Check if device can do manual exposure
        val caps = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManual = caps?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true

        val shutterFps = sharedPreferences.getString("shutter_speed", "15")?.toIntOrNull() ?: 15
        val shutterValueNs = if (shutterFps > 0) 1_000_000_000L / shutterFps else 0L

        if (!canManual || shutterValueNs <= 0L) {
            // full auto
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            return
        }

        // retrieve exposure & iso ranges
        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        if (exposureRange == null || isoRange == null) {
            // fallback to auto
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            return
        }

        // clamp shutter & pick safe ISO
        val safeExposure = shutterValueNs.coerceIn(exposureRange.lower, exposureRange.upper)
        val safeISO = max(isoRange.lower, 100)

        // set manual
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposure)
        builder.set(CaptureRequest.SENSOR_SENSITIVITY, safeISO)
    }

    /**
     * Detect phone orientation and set the orientation hint on the MediaRecorder.
     * This ensures the final MP4 is oriented correctly.
     */
    private fun setRecorderOrientation() {
        val rotation = mainActivity.windowManager.defaultDisplay.rotation
        // Swap each mapping if your device is ending up reversed
        val orientationHint = when (rotation) {
            Surface.ROTATION_0   -> 90 // was 90
            Surface.ROTATION_90  -> 180 // was 0
            Surface.ROTATION_180 -> 270  // was 270
            Surface.ROTATION_270 -> 0   // was 180
            else -> 0
        }
        mediaRecorder?.setOrientationHint(orientationHint)
    }
}
