package com.developer27.xemotion.inference

import android.app.Activity
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class TFLiteClassifier(activity: Activity) {

    companion object {
        private const val TAG = "TFLiteClassifier"
        private const val MODEL_FILENAME = "resnet18_custom.tflite"
        private const val LABELS_FILENAME = "labels.txt" // File in assets containing the labels
        private const val INPUT_SIZE = 224
        private const val BYTES_PER_CHANNEL = 4  // float32
        private const val NUM_THREADS = 4      // Adjust number of threads for performance

        // Normalization values (as used during training)
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }

    // Load labels from assets (one label per line).
    // If the order doesn't match your model, adjust accordingly.
    private val labels: List<String> = loadLabels(activity)
    // Example: If needed, uncomment the next line:
    // private val labels: List<String> = loadLabels(activity).reversed()

    // Create TFLite interpreter with options for improved performance.
    private var tflite: Interpreter = Interpreter(
        loadModelFile(activity, MODEL_FILENAME),
        Interpreter.Options().apply {
            setNumThreads(NUM_THREADS)
            // Optional: add GPU delegate or NNAPI delegate here if available.
        }
    )

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, modelFilename: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = activity.assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    private fun loadLabels(activity: Activity): List<String> {
        val labels = mutableListOf<String>()
        val reader = BufferedReader(InputStreamReader(activity.assets.open(LABELS_FILENAME)))
        var line: String?
        var isFirstLine = true
        while (reader.readLine().also { line = it } != null) {
            if (isFirstLine) {
                isFirstLine = false
                // If the first line looks like a package name, skip it.
                if (line!!.contains(".")) {
                    continue
                }
            }
            if (line!!.isNotBlank()) {
                labels.add(line!!.trim())
            }
        }
        reader.close()
        return labels
    }

    /**
     * Classifies the given Bitmap and returns a multi-line message showing:
     * - The raw output value for each class.
     * - The predicted label based on the highest output value.
     * - The overall confidence (computed as the softmax probability for the predicted label).
     */
    fun classifyImage(bitmap: Bitmap): String {
        // Resize the bitmap to the model's expected input size.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Prepare output array. Expected shape: [1, number of classes].
        val output = Array(1) { FloatArray(labels.size) }

        // Run inference.
        tflite.run(inputBuffer, output)

        // Get raw output values (logits).
        val rawOutput = output[0]

        // Determine the predicted index based on the raw output value.
        val predictedIndex = rawOutput.indices.maxByOrNull { rawOutput[it] } ?: -1
        val predictedLabel = labels.getOrElse(predictedIndex) { "Unknown" }

        // Compute softmax probabilities from the raw outputs.
        val expValues = rawOutput.map { exp(it.toDouble()) }
        val sumExp = expValues.sum()
        val predictedConfidence = if (sumExp != 0.0) {
            expValues[predictedIndex] / sumExp * 100
        } else {
            0.0
        }

        // Build the multi-line message.
        val message = buildString {
            append("Possible classes:\n")
            for (i in labels.indices) {
                append("${labels[i]}: ${rawOutput[i]}\n")
            }
            append("\nModel predicted: $predictedLabel (value: ${rawOutput[predictedIndex]})\n")
            append("Overall confidence: ${"%.2f".format(predictedConfidence)}%")
        }

        Log.d(TAG, message)
        return message
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(BYTES_PER_CHANNEL * INPUT_SIZE * INPUT_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Convert each pixel to normalized float values.
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Normalize using (value - mean) / std.
            byteBuffer.putFloat((r - MEAN_R) / STD_R)
            byteBuffer.putFloat((g - MEAN_G) / STD_G)
            byteBuffer.putFloat((b - MEAN_B) / STD_B)
        }
        return byteBuffer
    }

    fun close() {
        tflite.close()
    }
}