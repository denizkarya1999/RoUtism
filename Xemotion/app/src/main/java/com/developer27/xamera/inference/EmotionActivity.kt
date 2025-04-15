package com.developer27.xemotion.inference

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xemotion.R
import java.io.IOException

/**
 * EmotionActivity.kt
 *
 * This activity lets the user select one or more images, then prompts the user to guess a class name.
 * For each image whose predicted label matches the guessed class, the app displays its thumbnail along with
 * detailed features including predictions to other classes, the file name, overall confidence (accuracy), etc.
 */
class EmotionActivity : AppCompatActivity() {

    companion object {
        private const val PICK_FILE_REQUEST = 1
    }

    private lateinit var classifier: TFLiteClassifier
    private lateinit var resultTextView: TextView
    private lateinit var selectFileButton: Button
    private lateinit var resultsContainer: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emotion)

        // Get references from the layout.
        resultTextView = findViewById(R.id.resultTextView)
        selectFileButton = findViewById(R.id.selectImageButton)
        resultsContainer = findViewById(R.id.resultsContainer) // Container for dynamic matched image details.
        val titleContainer: LinearLayout = findViewById(R.id.title_container)

        // When clicking the title container, open the developer website.
        titleContainer.setOnClickListener {
            val url = "https://www.developer27.com"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        try {
            classifier = TFLiteClassifier(this)
        } catch (e: IOException) {
            e.printStackTrace()
            resultTextView.text = "Error initializing classifier"
        }

        selectFileButton.setOnClickListener {
            openFileChooser()
        }
    }

    /**
     * Opens the file chooser to select one or more images.
     */
    private fun openFileChooser() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"  // Restrict to images.
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)  // Allow multiple selection.
        startActivityForResult(intent, PICK_FILE_REQUEST)
    }

    /**
     * Handles the file chooser result. Images are stored along with their Uri so that file names can be extracted.
     * Then an AlertDialog prompts the user for a guess. For each image that is classified as the guessed class,
     * the app displays its thumbnail along with detailed features.
     */
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_FILE_REQUEST && resultCode == Activity.RESULT_OK) {
            // Store each selected image with its Uri for file name extraction.
            val selectedImages = mutableListOf<Pair<Bitmap, Uri>>()

            if (data?.clipData != null) {
                val clipData = data.clipData!!
                for (i in 0 until clipData.itemCount) {
                    val uri = clipData.getItemAt(i).uri
                    try {
                        val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                        selectedImages.add(Pair(bitmap, uri))
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                }
            } else if (data?.data != null) {
                val uri: Uri = data.data!!
                try {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                    selectedImages.add(Pair(bitmap, uri))
                } catch (e: IOException) {
                    e.printStackTrace()
                    resultTextView.text = "Failed to load image. Ensure you selected a valid image file."
                    return
                }
            }

            if (selectedImages.isEmpty()) {
                resultTextView.text = "No image selected."
                return
            }

            // Clear any previous results.
            resultsContainer.removeAllViews()

            // Prompt the user to guess a class name.
            val builder = AlertDialog.Builder(this)
            builder.setTitle("Guess a Class Name")
            val input = EditText(this).apply {
                hint = "Enter class name"
            }
            builder.setView(input)
            builder.setPositiveButton("OK") { dialog, which ->
                val guessedClass = input.text.toString().trim()

                // A simple local data class to hold matched image information.
                data class MatchedImage(
                    val bitmap: Bitmap,
                    val uri: Uri,
                    val confidence: Float,
                    val details: String,
                    val fileName: String
                )

                val matchedImages = mutableListOf<MatchedImage>()

                // For each image, check if its predicted label matches the guessed class.
                for ((bitmap, uri) in selectedImages) {
                    val predicted = classifier.predictLabel(bitmap)
                    if (predicted.equals(guessedClass, ignoreCase = true)) {
                        // Get the confidence for the guessed class. (Ensure this method is implemented in your classifier.)
                        val confidence = classifier.getConfidenceForClass(bitmap, guessedClass)
                        // Get full classification details.
                        val details = classifier.classifyImage(bitmap)
                        // Extract the file name from the Uri.
                        val fileName = getFileName(uri)
                        matchedImages.add(MatchedImage(bitmap, uri, confidence, details, fileName))
                    }
                }

                // Sort matched images in descending order by confidence.
                matchedImages.sortByDescending { it.confidence }
                resultsContainer.removeAllViews()

                // For each matched image, create a card-like layout aligned to the left.
                for (item in matchedImages) {
                    val cardContainer = LinearLayout(this).apply {
                        orientation = LinearLayout.VERTICAL
                        setPadding(dpToPx(16), dpToPx(16), dpToPx(16), dpToPx(16))
                        // Set blue background using hex value.
                        setBackgroundColor(Color.parseColor("#00274C"))
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply { setMargins(0, 0, 0, dpToPx(16)) }
                        // Ensure left alignment.
                        gravity = Gravity.START
                    }

                    // Create an ImageView for the image thumbnail. Limit the height to 200dp.
                    val imageView = ImageView(this).apply {
                        setImageBitmap(item.bitmap)
                        adjustViewBounds = true
                        scaleType = ImageView.ScaleType.CENTER_CROP
                        layoutParams = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            dpToPx(200)
                        )
                    }

                    // Build a formatted information string using concatenation.
                    val infoText =
                        "File Name: ${item.fileName}\n" +
                                "Predicted: $guessedClass (Confidence: ${"%.2f".format(item.confidence)}%)\n" +
                                "------------------------------\n" +
                                "Details:\n" +
                                item.details

                    // Create a TextView for details with maize text color, left-aligned.
                    val infoTextView = TextView(this).apply {
                        text = infoText
                        textSize = 16f
                        setTextColor(Color.parseColor("#FFCB05"))
                        gravity = Gravity.START
                        textAlignment = View.TEXT_ALIGNMENT_TEXT_START
                    }

                    // Add the thumbnail and detail TextView to the card.
                    cardContainer.addView(imageView)
                    cardContainer.addView(infoTextView)

                    // Add the card to the results container.
                    resultsContainer.addView(cardContainer)
                }

                // Display overall summary.
                resultTextView.text = "You guessed '$guessedClass'. Out of ${selectedImages.size} images, ${matchedImages.size} matched your guess."
            }
            builder.setNegativeButton("Cancel") { dialog, which ->
                dialog.cancel()
            }
            builder.show()
        }
    }

    /**
     * Helper function to convert dp to pixels.
     */
    private fun dpToPx(dp: Int): Int {
        val density = resources.displayMetrics.density
        return (dp * density + 0.5f).toInt()
    }

    /**
     * Extracts the file name from the provided Uri using the content resolver.
     */
    private fun getFileName(uri: Uri): String {
        var result: String? = null
        val cursor = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val index = it.getColumnIndex("_display_name")
                if (index != -1) {
                    result = it.getString(index)
                }
            }
        }
        return result ?: "Unknown"
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}