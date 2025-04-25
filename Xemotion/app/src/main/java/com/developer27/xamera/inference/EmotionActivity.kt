package com.developer27.xemotion.inference

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.developer27.xemotion.R
import com.developer27.xemotion.databinding.ActivityEmotionBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class EmotionActivity : AppCompatActivity() {

    private lateinit var binding: ActivityEmotionBinding
    private lateinit var classifier: TFLiteClassifier

    // Allows user to select multiple images
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

        // Initialize classifier
        classifier = TFLiteClassifier(this)

        // Hide optional single-image UI
        binding.imageView.visibility = View.GONE
        binding.resultTextView.visibility = View.GONE

        // Button: pick multiple images
        binding.selectImageButton.setOnClickListener {
            pickImages.launch(arrayOf("image/*"))
        }

        // When the title container is clicked, open the URL in a browser.
        binding.titleContainer.setOnClickListener {
            val url = "https://www.developer27.com"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }
    }

    override fun onDestroy() {
        // Release TFLite resources
        classifier.close()
        super.onDestroy()
    }

    /**
     * Run inference on multiple selected images
     */
    private fun runBatchInference(uris: List<Uri>) {
        // Clear old results
        binding.resultsContainer.removeAllViews()

        lifecycleScope.launch(Dispatchers.IO) {
            val bitmaps = mutableListOf<Bitmap>()
            try {
                for (uri in uris) {
                    // Attempt to persist read permission
                    try {
                        contentResolver.takePersistableUriPermission(
                            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                        )
                    } catch (_: SecurityException) {
                        // If we can't take it, ignore
                    }

                    // Decode to software-based bitmap
                    val source = ImageDecoder.createSource(contentResolver, uri)
                    val bmp = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                        // Optionally match the model's input size
                        decoder.setTargetSize(classifier.inputWidth, classifier.inputHeight)
                    }
                    bitmaps.add(bmp)
                }

                // Classify the entire batch
                val results = classifier.classifyBatch(bitmaps)

                withContext(Dispatchers.Main) {
                    showResults(uris, results)
                }
            } catch (e: IOException) {
                e.printStackTrace()
            } finally {
                // Recycle bitmaps
                bitmaps.forEach(Bitmap::recycle)
            }
        }
    }

    /**
     * Display each image's top label + confidence, then show a summary
     */
    private fun showResults(
        uris: List<Uri>,
        results: List<Pair<String, FloatArray>>  // second is "prob array"
    ) {
        // Our label set in TFLiteClassifier: ["Angry", "Anxiety", "Excitement", "Sadness"]
        // We'll find the best label from the 4 probabilities (after softmax).
        val yellow = ContextCompat.getColor(this, R.color.yellow)
        val total = results.size

        // 1) Add a header that says "Selected Image(s)" with an emoji
        if (total > 0) {
            binding.resultsContainer.addView(TextView(this).apply {
                text = "\uD83D\uDDBC\uFE0F Selected Image(s):"
                setTextColor(yellow)
                textSize = 18f
                setPadding(0, 8, 0, 12) // some spacing
            })
        }

        // We'll also track summary counts
        val labelCounts = mutableMapOf<String, Int>()

        // 2) For each image, pick the label with highest probability
        results.forEachIndexed { idx, (bestLabel, probArray) ->
            // Count the best label
            labelCounts[bestLabel] = (labelCounts[bestLabel] ?: 0) + 1

            // Format them to show each label's percentage
            val labelIndex = classifier.labels.indexOf(bestLabel)
            val bestConf = if (labelIndex in probArray.indices) probArray[labelIndex] else 0f
            val pctStr = String.format("%.2f", bestConf * 100f)

            val fileName = getFileName(uris[idx])
            binding.resultsContainer.addView(TextView(this).apply {
                text = "$fileName → $bestLabel ($pctStr%)"
                setTextColor(yellow)
                textSize = 16f
                setPadding(0, 8, 0, 8)
            })
        }

        // 3) Show summary with counts & percentages if more than 1 image
        if (total > 1) {
            binding.resultsContainer.addView(TextView(this).apply {
                text = "\n📋 Summary:"
                setTextColor(yellow)
                textSize = 18f
                setPadding(0, 12, 0, 4)
            })

            classifier.labels.forEach { lbl ->
                val cnt = labelCounts[lbl] ?: 0
                val pct = (cnt / total.toFloat()) * 100f
                binding.resultsContainer.addView(TextView(this).apply {
                    text = String.format(
                        "%d/%d (%.0f%%) images are %s",
                        cnt, total, pct, lbl
                    )
                    setTextColor(yellow)
                    textSize = 16f
                    setPadding(0, 4, 0, 4)
                })
            }
        }
    }

    private fun getFileName(uri: Uri): String {
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) {
                return c.getString(idx)
            }
        }
        return uri.lastPathSegment ?: "image"
    }
}