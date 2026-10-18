plugins {
    kotlin("jvm") version "2.2.10"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.json:json:20240303")
}

application {
    mainClass = "eval.AdviceRegressionProbeKt"
}
