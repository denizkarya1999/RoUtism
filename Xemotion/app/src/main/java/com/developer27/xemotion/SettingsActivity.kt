package com.developer27.xemotion

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)

        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    override fun onBackPressed() {
        super.onBackPressed()
        // Save settings and go back to MainActivity with result
        setResult(RESULT_OK, Intent())
        finish()
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // Listen for taps on our new preference
            findPreference<Preference>("key_launch_emotion_activity")?.setOnPreferenceClickListener {
                // Launch the EmotionActivity
                startActivity(Intent(requireContext(), com.developer27.xemotion.inference.EmotionActivity::class.java))
                true
            }
        }
    }
}