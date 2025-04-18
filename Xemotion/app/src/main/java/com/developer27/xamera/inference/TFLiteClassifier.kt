package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

class TFLiteClassifier(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    val inputWidth: Int
    val inputHeight: Int
    private val nchw: Boolean

    // Use the same 4-class set from your Python code
    // Order must match model's output indexes
    val labels = listOf("Angry", "Anxiety", "Excitement", "Sadness")

    init {
        // Load your TFLite model from assets
        val fd = context.assets.openFd("resnet50_emotion.tflite")
        interpreter = FileInputStream(fd.fileDescriptor).channel.use { ch ->
            val mapped = ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            Interpreter(mapped)
        }

        // Check input shape. E.g. [1,3,H,W] or [1,H,W,3].
        val shape = interpreter.getInputTensor(0).shape()
        nchw        = (shape.size == 4 && shape[1] == 3)
        inputHeight = if (nchw) shape[2] else shape[1]
        inputWidth  = if (nchw) shape[3] else shape[2]
    }

    /**
     * Classify a batch of bitmaps.
     * Returns a List of (bestLabel, probDistribution), where:
     *   - bestLabel is a String, the highest-confidence label
     *   - probDistribution is a FloatArray of size=4 (after softmax)
     */
    fun classifyBatch(bitmaps: List<Bitmap>): List<Pair<String, FloatArray>> {
        require(bitmaps.isNotEmpty()) { "Bitmap list is empty." }

        // Possibly resize the interpreter's input to match batch size
        val batchSize = bitmaps.size
        val inShape = interpreter.getInputTensor(0).shape()  // e.g. [1,3,H,W]
        if (inShape[0] != batchSize) {
            interpreter.resizeInput(0, intArrayOf(batchSize, *inShape.copyOfRange(1, 4)))
            interpreter.allocateTensors()
        }

        // Prepare a ByteBuffer for all images
        val imgSize = inputWidth * inputHeight * 3L
        require(4L * imgSize * batchSize <= Int.MAX_VALUE) {
            "Input too large – consider fewer images."
        }
        val inputBuffer = ByteBuffer.allocateDirect((4L * imgSize * batchSize).toInt())
            .order(ByteOrder.nativeOrder())

        // Example normalization for ImageNet-like models
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std  = floatArrayOf(0.229f, 0.224f, 0.225f)
        val pixels = IntArray(inputWidth * inputHeight)

        // Helper to normalize each channel
        fun norm(c: Int, ch: Int): Float {
            // c >> 16 is R, c >> 8 is G, c & 0xFF is B
            val channelVal = ((c shr (16 - 8*ch)) and 0xFF) / 255f
            return (channelVal - mean[ch]) / std[ch]
        }

        // Fill the buffer with pixel data
        for (bmp in bitmaps) {
            bmp.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

            if (nchw) {
                // Channel-first layout
                for (ch in 0..2) {
                    for (px in pixels) {
                        inputBuffer.putFloat(norm(px, ch))
                    }
                }
            } else {
                // Channel-last layout
                for (px in pixels) {
                    inputBuffer.putFloat(norm(px, 0))
                    inputBuffer.putFloat(norm(px, 1))
                    inputBuffer.putFloat(norm(px, 2))
                }
            }
        }
        inputBuffer.rewind()

        // Prepare output array; typically shape [batchSize, 4] for 4 classes
        val rawOutput = Array(batchSize) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, rawOutput)

        // Convert each row (logits) to probabilities via softmax
        return rawOutput.map { logits ->
            val probabilities = softmax(logits)
            val bestIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            labels[bestIndex] to probabilities
        }
    }

    /**
     * Compute softmax for a vector of logits.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        // Subtract max for numerical stability
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { (it / sumExps) }.toFloatArray()
    }

    override fun close() {
        interpreter.close()
    }
}
