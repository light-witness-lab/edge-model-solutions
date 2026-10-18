plugins {
    kotlin("jvm") version "2.2.10"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime:1.22.0")
    implementation("org.json:json:20240303")
}

application {
    mainClass = "verify.MainKt"
}
