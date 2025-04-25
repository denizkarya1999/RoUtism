package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * A unified TFLiteClassifier that can:
 *  1) Classify multiple images (batch) for emotion detection.
 *  2) Classify a single "line" bitmap in a function called classifyLine().
 *
 * You can load the same TFLite model for both tasks if they share the same architecture
 * and input dimension. Otherwise, consider having separate classifier instances or
 * separate label lists.
 */
class TFLiteClassifier(
    private val context: Context,
    // Name of your model asset file. E.g. "resnet50_emotion.tflite" or "line_classifier.tflite"
    private val modelFileName: String = "resnet50_emotion.tflite"
) : AutoCloseable {

    // The TensorFlow Lite interpreter
    private val interpreter: Interpreter

    // Assume the model’s input shape: [1, height, width, 3] or [1, 3, height, width].
    // We'll parse it dynamically in init{} below.
    val inputWidth: Int
    val inputHeight: Int
    private val nchw: Boolean  // Whether the model expects [1, 3, H, W] (NCHW)

    // A label list. Adjust if your model outputs different classes.
    // For line classification, you might have different categories.
    val labels = listOf("Angry", "Anxiety", "Excitement", "Sadness")

    init {
        // Load the TFLite model as a MappedByteBuffer from assets
        val tfliteModel = loadModelFileFromAssets(modelFileName)
        interpreter = Interpreter(tfliteModel)

        // Check input shape, e.g. [1, 3, 224, 224] or [1, 224, 224, 3]
        val shape = interpreter.getInputTensor(0).shape()
        nchw = (shape.size == 4 && shape[1] == 3)

        // Extract input dimensions
        inputHeight = if (nchw) shape[2] else shape[1]
        inputWidth  = if (nchw) shape[3] else shape[2]
    }

    /**
     * Classify a *batch* of images for emotion detection (multi-image).
     * Returns a List of (bestLabel, allProbabilities),
     * where bestLabel is a String, and allProbabilities is FloatArray of size=labels.size.
     */
    fun classifyBatch(bitmaps: List<Bitmap>): List<Pair<String, FloatArray>> {
        require(bitmaps.isNotEmpty()) { "Bitmap list is empty." }

        val batchSize = bitmaps.size
        // Possibly resize interpreter input to [batchSize, 3, H, W] or [batchSize, H, W, 3]
        val inShape = interpreter.getInputTensor(0).shape()  // e.g. [1, 3, 224, 224]
        if (inShape[0] != batchSize) {
            interpreter.resizeInput(0, intArrayOf(batchSize, *inShape.copyOfRange(1, 4)))
            interpreter.allocateTensors()
        }

        // Create the input buffer for all images
        val imgSize = inputWidth * inputHeight * 3L
        require(4L * imgSize * batchSize <= Int.MAX_VALUE) {
            "Input too large; consider smaller batch or smaller resolution."
        }
        val inputBuffer = ByteBuffer.allocateDirect((4L * imgSize * batchSize).toInt())
            .order(ByteOrder.nativeOrder())

        // Fill the buffer with pixel data from each bitmap
        for (bmp in bitmaps) {
            writeImageToBuffer(bmp, inputBuffer)
        }
        inputBuffer.rewind()

        // Prepare output array: typically [batchSize, #labels]
        val rawOutput = Array(batchSize) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, rawOutput)

        // Convert each row (logits) to probabilities via softmax
        return rawOutput.map { logits ->
            val probs = softmax(logits)
            val bestIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
            labels[bestIndex] to probs
        }
    }

    /**
     * Classify a SINGLE "line" bitmap.
     *
     * 1) We resize the bitmap to the model's [inputWidth, inputHeight].
     * 2) Create an input buffer for a single image (batchSize=1).
     * 3) Collect the output probabilities.
     * 4) Return the best label with confidence, e.g. "Excitement (95.3%)".
     */
    fun classifyLine(lineBitmap: Bitmap): String {
        // We'll do a single-image approach
        // Make sure the interpreter is sized for batch=1
        val inShape = interpreter.getInputTensor(0).shape()  // e.g. [N, 3, H, W] or [N, H, W, 3]
        val currentBatch = inShape[0]

        if (currentBatch != 1) {
            // Force resize to [1, 3, H, W] or [1, H, W, 3]
            interpreter.resizeInput(0, intArrayOf(1, *inShape.copyOfRange(1, 4)))
            interpreter.allocateTensors()
        }

        // 1) Scale to match the model's required size
        val scaled = Bitmap.createScaledBitmap(lineBitmap, inputWidth, inputHeight, true)

        // 2) Prepare a buffer with just 1 image
        val imgSize = inputWidth * inputHeight * 3
        val buffer = ByteBuffer.allocateDirect(imgSize * 4).order(ByteOrder.nativeOrder())

        // 3) Fill in the pixel data
        writeImageToBuffer(scaled, buffer)
        buffer.rewind()

        // 4) Allocate output [1, #labels]
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(buffer, output)

        // 5) Softmax + pick best
        val probs = softmax(output[0])
        val bestIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val bestConf = probs[bestIndex]
        val bestLabel = labels[bestIndex]

        return "$bestLabel (${String.format("%.1f", bestConf * 100f)}%)"
    }

    /**
     * Helper: Write a bitmap's pixels into the model's input buffer,
     * respecting (NCHW vs NHWC) shape, and normalizing e.g. with ImageNet stats.
     * Adjust if your line model uses a simpler approach (like [0..1] w/o mean/std).
     */
    private fun writeImageToBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        // Example normalization for a typical ImageNet model
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        if (nchw) {
            // If model expects channel-first: [1,3,H,W]
            // We fill in all R for the entire image, then G, then B.
            for (c in 0..2) {
                for (py in 0 until height) {
                    for (px in 0 until width) {
                        val rgb = pixels[py * width + px]
                        val value = channelValue(rgb, c)
                        // Normalize
                        val normVal = (value - mean[c]) / std[c]
                        buffer.putFloat(normVal)
                    }
                }
            }
        } else {
            // Channel-last: [1,H,W,3]
            for (py in 0 until height) {
                for (px in 0 until width) {
                    val rgb = pixels[py * width + px]
                    // R
                    val rVal = channelValue(rgb, 0)
                    buffer.putFloat((rVal - mean[0]) / std[0])
                    // G
                    val gVal = channelValue(rgb, 1)
                    buffer.putFloat((gVal - mean[1]) / std[1])
                    // B
                    val bVal = channelValue(rgb, 2)
                    buffer.putFloat((bVal - mean[2]) / std[2])
                }
            }
        }
    }

    /**
     * Extract one channel [0: R, 1: G, 2: B] from an ARGB pixel, scaled to [0..1].
     */
    private fun channelValue(pixel: Int, channelIdx: Int): Float {
        return when(channelIdx) {
            0 -> ((pixel shr 16) and 0xFF) / 255f  // R
            1 -> ((pixel shr  8) and 0xFF) / 255f  // G
            else -> (pixel and 0xFF) / 255f        // B
        }
    }

    /**
     * Basic softmax function.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    /**
     * Load a TFLite model from assets folder into a MappedByteBuffer.
     */
    @Throws(IOException::class)
    private fun loadModelFileFromAssets(fileName: String): ByteBuffer {
        context.assets.openFd(fileName).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use { ch ->
                val startOffset = fd.startOffset
                val declaredLength = fd.declaredLength
                return ch.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            }
        }
    }

    /**
     * Closes the TFLite interpreter.
     */
    override fun close() {
        interpreter.close()
    }
}