pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        flatDir {
            dirs("${rootDir}/app/libs")
        }
    }
}

rootProject.name = "Xemotion"
include(":app")
include(":OpenCV-4.10.0")
