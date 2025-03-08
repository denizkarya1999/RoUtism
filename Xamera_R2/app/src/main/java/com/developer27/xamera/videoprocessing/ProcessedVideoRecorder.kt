package com.developer27.xamera.routism.videoprocessing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import android.view.Surface

/**
 * A helper class that encodes processed Bitmap frames to an H.264 video
 * and muxes them into an MP4 file.
 *
 * IMPORTANT: This sample assumes that the MediaCodec input Surface supports Canvas drawing.
 * In some cases you may need to use an EGL-based InputSurface.
 */
class ProcessedVideoRecorder(
    // Fixed resolution values known to work with MediaCodec (e.g., 416x416).
    private val width: Int,
    private val height: Int,
    private val outputPath: String,
    private val frameRate: Int = 30,
    // Instead of a fixed bit rate, compute one based on resolution.
    // For example, a multiplier of 5 gives: bitRate = width * height * 5.
    private val bitRateMultiplier: Int = 5
) {

    companion object {
        private const val TAG = "ProcessedVideoRecorder"
        private const val MIME_TYPE = "video/avc" // H.264 Advanced Video Coding
    }

    private var mediaCodec: MediaCodec? = null
    private var inputSurface: Surface? = null
    private var mediaMuxer: MediaMuxer? = null
    private var trackIndex: Int = -1
    private var muxerStarted = false
    private var frameCount = 0

    /**
     * Call this method when you wish to start recording.
     */
    fun start() {
        // Configure the video format with fixed resolution and computed bit rate.
        val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height)
        format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, width * height)
        // Dynamically compute bit rate.
        val calculatedBitRate = width * height * bitRateMultiplier
        format.setInteger(MediaFormat.KEY_BIT_RATE, calculatedBitRate)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1) // I-Frame every second

        try {
            mediaCodec = MediaCodec.createEncoderByType(MIME_TYPE)
        } catch (e: Exception) {
            throw RuntimeException("Unable to create encoder", e)
        }

        mediaCodec?.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        // Create the input surface for drawing processed frames.
        inputSurface = mediaCodec?.createInputSurface()
        mediaCodec?.start()

        // Create the MediaMuxer to write the output MP4 file.
        mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxerStarted = false
        frameCount = 0
        Log.d(TAG, "Recorder started with resolution: ${width}x${height}, Bitrate: $calculatedBitRate, output: $outputPath")
    }

    /**
     * Call this method to record a processed frame.
     * This method should be called once per frame.
     */
    fun recordFrame(bitmap: Bitmap) {
        // Draw the bitmap into the encoder’s input surface.
        val canvas: Canvas? = try {
            inputSurface?.lockCanvas(null)
        } catch (e: Exception) {
            Log.e(TAG, "Error locking canvas: ${e.message}")
            null
        }
        if (canvas != null) {
            // Scale the bitmap to the canvas dimensions.
            canvas.drawBitmap(bitmap, null, Rect(0, 0, canvas.width, canvas.height), null)
            inputSurface?.unlockCanvasAndPost(canvas)
        } else {
            Log.w(TAG, "Canvas was null; frame not recorded.")
        }

        // Drain the encoder's output to retrieve encoded frames.
        drainEncoder(endOfStream = false)
        frameCount++
    }

    /**
     * Call this method to stop recording. It finalizes the output file.
     */
    fun stop() {
        // Signal end-of-stream.
        drainEncoder(endOfStream = true)
        try {
            mediaCodec?.stop()
            mediaCodec?.release()
            mediaCodec = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping encoder: ${e.message}")
        }
        try {
            mediaMuxer?.stop()
            mediaMuxer?.release()
            mediaMuxer = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping muxer: ${e.message}")
        }
        Log.d(TAG, "Recording stopped. Total frames: $frameCount")
    }

    /**
     * Drains any available output from the encoder.
     * If endOfStream is true, signals the encoder that no more frames are coming.
     */
    private fun drainEncoder(endOfStream: Boolean) {
        if (endOfStream) {
            // Inform the encoder no more input frames will be provided.
            mediaCodec?.signalEndOfInputStream()
        }

        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferIndex = mediaCodec?.dequeueOutputBuffer(bufferInfo, 10000L) ?: break

            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) {
                        break // No output available yet.
                    }
                }
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) {
                        throw RuntimeException("Format changed twice")
                    }
                    val newFormat = mediaCodec?.outputFormat
                    trackIndex = mediaMuxer?.addTrack(newFormat!!)
                        ?: throw RuntimeException("Unable to add track to muxer")
                    mediaMuxer?.start()
                    muxerStarted = true
                }
                outputBufferIndex >= 0 -> {
                    val encodedData = mediaCodec?.getOutputBuffer(outputBufferIndex)
                        ?: throw RuntimeException("Encoder output buffer was null")
                    if (bufferInfo.size != 0) {
                        if (!muxerStarted) {
                            throw RuntimeException("Muxer hasn't started")
                        }
                        // Adjust buffer positions.
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        mediaMuxer?.writeSampleData(trackIndex, encodedData, bufferInfo)
                    }
                    mediaCodec?.releaseOutputBuffer(outputBufferIndex, false)
                    // Exit loop if end-of-stream flag is reached.
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break
                    }
                }
            }
        }
    }
}