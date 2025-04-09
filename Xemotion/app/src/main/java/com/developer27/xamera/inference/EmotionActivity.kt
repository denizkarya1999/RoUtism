package com.developer27.xemotion.inference

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xemotion.R
import java.io.IOException

class EmotionActivity : AppCompatActivity() {

    companion object {
        private const val PICK_FILE_REQUEST = 1
    }

    private lateinit var classifier: TFLiteClassifier
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var selectFileButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emotion) // Make sure this layout is centered as desired

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        selectFileButton = findViewById(R.id.selectImageButton)
        val titleContainer: LinearLayout = findViewById(R.id.title_container)

        // When the title container is clicked, open the URL in a browser.
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

    private fun openFileChooser() {
        // Open file explorer with no filter.
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "*/*"
        startActivityForResult(intent, PICK_FILE_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_FILE_REQUEST && resultCode == Activity.RESULT_OK && data?.data != null) {
            val fileUri: Uri = data.data!!
            try {
                // Decode the selected file as a Bitmap image.
                val bitmap: Bitmap = MediaStore.Images.Media.getBitmap(contentResolver, fileUri)
                imageView.setImageBitmap(bitmap)

                // Run inference on the image. The classifier returns a multi-line message.
                val message = classifier.classifyImage(bitmap)

                // Write the message directly to the label.
                resultTextView.text = message

            } catch (e: IOException) {
                e.printStackTrace()
                resultTextView.text = "Failed to load image. Ensure you selected an image file."
            } catch (e: Exception) {
                e.printStackTrace()
                resultTextView.text = "Selected file is not a valid image."
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}