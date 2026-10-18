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
    // Frozen prompt probe stays the default entry; T36 selects the resolution probe via
    // -PmainClass=verify.ResolutionProbeKt without changing the default reproduce command.
    mainClass = (findProperty("mainClass") as String?) ?: "verify.MainKt"
}
