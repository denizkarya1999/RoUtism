package com.developer27.xemotion.inference

import android.app.Activity
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class TFLiteClassifier(activity: Activity) {

    companion object {
        private const val TAG = "TFLiteClassifier"
        private const val MODEL_FILENAME = "resnet50_custom.tflite"
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

    // Load labels from assets (one label per line)
    private val labels: List<String> = loadLabels(activity)

    // Create TFLite interpreter with options for improved performance.
    private var tflite: Interpreter = Interpreter(
        loadModelFile(activity, MODEL_FILENAME),
        Interpreter.Options().apply {
            setNumThreads(NUM_THREADS)
            // Optionally add GPU/NNAPI delegates here if available.
        }
    )

    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, modelFilename: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = activity.assets.openFd(modelFilename)
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    private fun loadLabels(activity: Activity): List<String> {
        val labels = mutableListOf<String>()
        activity.assets.open(LABELS_FILENAME).bufferedReader().use { reader ->
            var firstLine = true
            reader.forEachLine { currentLine ->
                // Skip the first line if it appears to be a package name.
                if (firstLine) {
                    firstLine = false
                    if (currentLine.contains(".")) {
                        return@forEachLine  // Skip this line.
                    }
                }
                if (currentLine.isNotBlank()) {
                    labels.add(currentLine.trim())
                }
            }
        }
        return labels
    }

    /**
     * Classifies a single Bitmap image and returns a multi-line message with the results.
     */
    fun classifyImage(bitmap: Bitmap): String {
        // Resize the bitmap to the expected input size.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Prepare output array with shape [1, number of classes].
        val output = Array(1) { FloatArray(labels.size) }

        // Run inference.
        tflite.run(inputBuffer, output)
        val rawOutput = output[0]

        // Find the predicted class index.
        val predictedIndex = rawOutput.indices.maxByOrNull { rawOutput[it] } ?: -1
        val predictedLabel = labels.getOrElse(predictedIndex) { "Unknown" }

        // Compute softmax confidence.
        val expValues = rawOutput.map { exp(it.toDouble()) }
        val sumExp = expValues.sum()
        val predictedConfidence = if (sumExp != 0.0) {
            expValues[predictedIndex] / sumExp * 100
        } else {
            0.0
        }

        return buildString {
            append("Possible classes:\n")
            for (i in labels.indices) {
                append("${labels[i]}: ${rawOutput[i]}\n")
            }
            append("\nModel predicted: $predictedLabel (value: ${rawOutput[predictedIndex]})\n")
            append("Overall confidence: ${"%.2f".format(predictedConfidence)}%")
        }
    }

    /**
     * Classifies a list of Bitmap images by averaging their raw outputs and returning one overall prediction.
     *
     * @param bitmaps A list of images to classify.
     * @return A message containing the averaged prediction across all images.
     */
    fun classifyImagesAverage(bitmaps: List<Bitmap>): String {
        val batchSize = bitmaps.size
        // Calculate the required buffer size for all images.
        val imageByteCount = INPUT_SIZE * INPUT_SIZE * 3 * BYTES_PER_CHANNEL
        val batchByteCount = batchSize * imageByteCount

        // Allocate a single ByteBuffer to hold the batch.
        val inputBuffer = ByteBuffer.allocateDirect(batchByteCount)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert each bitmap to the expected input format and write into the batch buffer.
        for (bitmap in bitmaps) {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
            for (pixel in pixels) {
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                // Normalize each pixel value.
                inputBuffer.putFloat((r - MEAN_R) / STD_R)
                inputBuffer.putFloat((g - MEAN_G) / STD_G)
                inputBuffer.putFloat((b - MEAN_B) / STD_B)
            }
        }
        inputBuffer.rewind()

        // Prepare output array with shape [batchSize, number of classes].
        val output = Array(batchSize) { FloatArray(labels.size) }

        // Run inference on the entire batch.
        tflite.run(inputBuffer, output)

        // Average the predictions across all images (per class).
        val avgOutput = FloatArray(labels.size) { 0f }
        for (i in 0 until batchSize) {
            for (j in labels.indices) {
                avgOutput[j] += output[i][j]
            }
        }
        for (j in labels.indices) {
            avgOutput[j] /= batchSize.toFloat()
        }

        // Determine the predicted label from the averaged outputs.
        val predictedIndex = avgOutput.indices.maxByOrNull { avgOutput[it] } ?: -1
        val predictedLabel = labels.getOrElse(predictedIndex) { "Unknown" }

        // Compute the softmax probability from the averaged outputs.
        val expValues = avgOutput.map { exp(it.toDouble()) }
        val sumExp = expValues.sum()
        val predictedConfidence = if (sumExp != 0.0) {
            expValues[predictedIndex] / sumExp * 100
        } else {
            0.0
        }

        return buildString {
            append("Based on the average prediction from $batchSize images:\n")
            append("Averaged raw outputs:\n")
            for (j in labels.indices) {
                append("${labels[j]}: ${avgOutput[j]}\n")
            }
            append("\nModel predicted: $predictedLabel (value: ${avgOutput[predictedIndex]})\n")
            append("Overall confidence: ${"%.2f".format(predictedConfidence)}%")
        }
    }

    fun getConfidenceForClass(bitmap: Bitmap, className: String): Float {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)
        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(inputBuffer, output)
        val rawOutput = output[0]
        val expValues = rawOutput.map { exp(it.toDouble()) }
        val sumExp = expValues.sum()
        val index = labels.indexOfFirst { it.equals(className, ignoreCase = true) }
        if (index != -1 && sumExp != 0.0) {
            return (expValues[index] / sumExp * 100).toFloat()
        }
        return 0f
    }

    /**
     * Converts a Bitmap to a ByteBuffer in a format suitable for model input.
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(BYTES_PER_CHANNEL * INPUT_SIZE * INPUT_SIZE * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            byteBuffer.putFloat((r - MEAN_R) / STD_R)
            byteBuffer.putFloat((g - MEAN_G) / STD_G)
            byteBuffer.putFloat((b - MEAN_B) / STD_B)
        }
        return byteBuffer
    }

    /**
     * Releases resources associated with the TFLite interpreter.
     */
    fun close() {
        tflite.close()
    }

    /**
     * Performs inference on the given Bitmap image and returns the predicted label.
     */
    fun predictLabel(bitmap: Bitmap): String {
        // Resize the bitmap to the expected input size.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Prepare output array with shape [1, number of classes].
        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(inputBuffer, output)
        val rawOutput = output[0]

        // Find the index of the highest output value.
        val predictedIndex = rawOutput.indices.maxByOrNull { rawOutput[it] } ?: -1
        return labels.getOrElse(predictedIndex) { "Unknown" }
    }
}
