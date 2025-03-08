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

rootProject.name = "Xamera R2"
include(":app")
include(":OpenCV-4.10.0")
