// File: EmotionActivity.kt
package com.developer27.xemotion.inference

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.developer27.xemotion.R
import com.developer27.xemotion.databinding.ActivityEmotionBinding
import com.developer27.xemotion.videoprocessing.VideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class EmotionActivity : AppCompatActivity() {

    private lateinit var binding: ActivityEmotionBinding
    private lateinit var classifier: PyTorchClassifier
    private lateinit var videoProcessor: VideoProcessor
    private lateinit var resultImageView: ImageView

    private val pickImages = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris: List<Uri>? ->
        if (!uris.isNullOrEmpty()) {
            runBatchInference(uris)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityEmotionBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 1) Initialize classifier
        classifier = PyTorchClassifier(this)

        // 2) Initialize VideoProcessor
        videoProcessor = VideoProcessor(this)

        // 3) Grab the overlay ImageView
        resultImageView = binding.resultImageView

        // 4) Hide optional UI elements
        binding.imageView.visibility = View.GONE
        binding.resultTextView.visibility = View.GONE
        resultImageView.visibility = View.GONE

        // 5) Button: pick multiple images
        binding.selectImageButton.setOnClickListener {
            pickImages.launch(arrayOf("image/*"))
        }

        // 6) Title container -> open website
        binding.titleContainer.setOnClickListener {
            val url = "https://www.developer27.com"
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        }
    }

    override fun onDestroy() {
        classifier.close()
        super.onDestroy()
    }

    private fun runBatchInference(uris: List<Uri>) {
        binding.resultsContainer.removeAllViews()
        binding.resultTextView.visibility = View.GONE
        resultImageView.visibility = View.GONE

        // Load + classify images in background
        lifecycleScope.launch(Dispatchers.IO) {
            val bitmaps = mutableListOf<Bitmap>()
            try {
                for (uri in uris) {
                    try {
                        // In case the system wants to persist read permission
                        contentResolver.takePersistableUriPermission(
                            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                        )
                    } catch (_: SecurityException) { }
                    val source = ImageDecoder.createSource(contentResolver, uri)
                    val bmp = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        // optionally set a target size close to the classifier input
                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                        decoder.setTargetSize(classifier.inputWidth, classifier.inputHeight)
                    }
                    bitmaps.add(bmp)
                }

                // Classify them
                val results = classifier.classifyBatch(bitmaps)

                withContext(Dispatchers.Main) {
                    showResults(uris, bitmaps, results)
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    private fun showResults(
        uris: List<Uri>,
        bitmaps: List<Bitmap>,
        results: List<Pair<String, FloatArray>>
    ) {
        val yellow = ContextCompat.getColor(this, R.color.yellow)
        val total = results.size

        if (total > 1) {
            // Multiple images
            binding.resultsContainer.addView(TextView(this).apply {
                text = "\uD83D\uDDBC\uFE0F Selected Image(s):"
                setTextColor(yellow)
                textSize = 18f
                setPadding(0, 8, 0, 12)
            })
            val labelCounts = mutableMapOf<String, Int>()

            results.forEachIndexed { idx, (_, probArray) ->
                // compute top-3
                val top3 = probArray
                    .withIndex()
                    .sortedByDescending { it.value }
                    .take(3)
                    .map { (i, p) -> classifier.labels[i] to p * 100f }

                // count top-1 for summary
                labelCounts[top3.first().first] = (labelCounts[top3.first().first] ?: 0) + 1

                val fileName = getFileName(uris[idx])

                // row
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, 8, 0, 8)
                }

                // vertical info: file + top3
                val infoCol = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                }
                infoCol.addView(TextView(this).apply {
                    text = "$fileName:"
                    setTextColor(yellow)
                    textSize = 16f
                    setPadding(0, 0, 0, 4)
                })
                top3.forEach { (lbl, pct) ->
                    infoCol.addView(TextView(this).apply {
                        text = "• $lbl (${String.format("%.2f", pct)}%)"
                        setTextColor(yellow)
                        textSize = 14f
                        setPadding(16, 2, 0, 2)
                    })
                }
                row.addView(infoCol)

                // generate overlay
                val overlay = processFrameForOverlay(bitmaps[idx])
                if (overlay != null) {
                    row.addView(ImageView(this).apply {
                        val dp = 200
                        val px = (dp * resources.displayMetrics.density).toInt()
                        layoutParams = LinearLayout.LayoutParams(px, px).apply {
                            marginStart = 16
                        }
                        adjustViewBounds = true
                        scaleType = ImageView.ScaleType.CENTER_CROP
                        setImageBitmap(overlay)
                    })
                }

                binding.resultsContainer.addView(row)
            }

            // summary of top-1
            binding.resultsContainer.addView(TextView(this).apply {
                text = "\n📋 Summary of top-1 predictions:"
                setTextColor(yellow)
                textSize = 18f
                setPadding(0, 12, 0, 4)
            })
            classifier.labels.forEach { lbl ->
                val cnt = labelCounts[lbl] ?: 0
                val pct = (cnt / total.toFloat()) * 100f
                binding.resultsContainer.addView(TextView(this).apply {
                    text = String.format("%d/%d (%.0f%%) are %s", cnt, total, pct, lbl)
                    setTextColor(yellow)
                    textSize = 16f
                    setPadding(0, 4, 0, 4)
                })
            }

        } else if (total == 1) {
            // single-image
            val (_, probArray) = results[0]
            val top3 = probArray
                .withIndex()
                .sortedByDescending { it.value }
                .take(3)
                .map { (i, p) -> classifier.labels[i] to p * 100f }
            val fileName = getFileName(uris[0])

            binding.resultTextView.text = buildString {
                append("$fileName:\n")
                top3.forEach { (lbl, pct) ->
                    append("• $lbl (${String.format("%.2f", pct)}%)\n")
                }
            }
            binding.resultTextView.visibility = View.VISIBLE

            val overlay = processFrameForOverlay(bitmaps[0])
            if (overlay != null) {
                resultImageView.setImageBitmap(overlay)
                resultImageView.visibility = View.VISIBLE
            }
        }
    }

    /**
     * Example: uses the synchronous method in VideoProcessor to get an overlay.
     */
    private fun processFrameForOverlay(bitmap: Bitmap): Bitmap? {
        // If you always want YOLO or contour detection, set the detection mode accordingly.
        // Or you can change Settings.DetectionMode.current before calling this.
        // For now, we just do a synchronous call:
        return videoProcessor.processFrameSynchronous(bitmap)?.first
    }

    private fun getFileName(uri: Uri): String =
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) return c.getString(idx)
            null
        } ?: uri.lastPathSegment.orEmpty()
}