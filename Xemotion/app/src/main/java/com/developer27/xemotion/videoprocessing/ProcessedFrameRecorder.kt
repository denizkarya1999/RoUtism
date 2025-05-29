package com.developer27.xemotion.videoprocessing

import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ProcessedFrameRecorder(
    private val outputPath: String,
    private val format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG,
    private val quality: Int = 100 // Quality range: 0 (lowest) to 100 (highest)
) {
    companion object {
        private const val TAG = "ProcessedFrameSaver"
    }

    fun save(bitmap: Bitmap): Boolean {
        var outputStream: FileOutputStream? = null
        return try {
            val file = File(outputPath)
            // Create parent directories if they do not exist.
            file.parentFile?.mkdirs()

            outputStream = FileOutputStream(file)
            val successful = bitmap.compress(format, quality, outputStream)
            outputStream.flush()
            if (successful) {
                Log.d(TAG, "Image saved successfully to $outputPath")
            } else {
                Log.e(TAG, "Bitmap.compress() returned false. Image not saved.")
            }
            successful
        } catch (e: IOException) {
            Log.e(TAG, "Error saving image: ${e.message}")
            false
        } finally {
            try {
                outputStream?.close()
            } catch (e: IOException) {
                Log.e(TAG, "Error closing output stream: ${e.message}")
            }
        }
    }
}