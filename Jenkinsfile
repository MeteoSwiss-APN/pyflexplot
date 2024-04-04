class Globals {
    // constants
    static final String PROJECT = 'pyflexplot'
    static final String IMAGE_REPO = 'docker-intern-nexus.meteoswiss.ch'
    static final String IMAGE_NAME = 'docker-intern-nexus.meteoswiss.ch/flexpart-cosmo/pyflexplot'

    // sets the pipeline to execute all steps related to building the service
    static boolean build = false

    // sets to abort the pipeline if the Sonarqube QualityGate fails
    static boolean qualityGateAbortPipeline = false

    // sets the pipeline to execute all steps related to releasing the service
    static boolean release = false

    // sets the pipeline to execute all steps related to deployment of the service
    static boolean deploy = false

    // sets the pipeline to execute all steps related to restart the service
    static boolean restart = false

    // sets the pipeline to execute all steps related to delete the service from the container platform
    static boolean deleteContainer = false

    // sets the pipeline to execute all steps related to trigger the trivy scan
    static boolean runTrivyScan = false

    // the image tag used for tagging the image
    static String imageTag = ''

    // the service version
    static String version = ''
}

// TODO RMF-81 add documentation stage

@Library('dev_tools@main') _
pipeline {
    agent { label 'podman' }

    parameters {
        choice(choices: ['Build', 'Deploy', 'Release', 'Restart', 'Delete', 'Trivy-Scan'],
               description: 'Build type',
               name: 'buildChoice')

        choice(choices: ['flexpart-cosmo-devt'],
               description: 'Environment',
               name: 'environment')
    }

    options {
        // New jobs should wait until older jobs are finished
        disableConcurrentBuilds()
        // Discard old builds
        buildDiscarder(logRotator(artifactDaysToKeepStr: '7', artifactNumToKeepStr: '1', daysToKeepStr: '45', numToKeepStr: '10'))
        // Timeout the pipeline build after 1 hour
        timeout(time: 1, unit: 'HOURS')
    }

    environment {
        scannerHome = tool name: 'Sonarqube-certs-PROD', type: 'hudson.plugins.sonar.SonarRunnerInstallation'
    }

    stages {
        stage('Preflight') {
            steps {
                script {
                    echo 'Starting with Preflight'

                    // Determine the type of build
                    switch (params.buildChoice) {
                        case 'Build':
                            Globals.build = true
                            break
                        case 'Deploy':
                            Globals.deploy = true
                            break
                        case 'Release':
                            Globals.release = true
                            Globals.build = true
                            break
                        case 'Restart':
                            Globals.restart = true
                            break
                        case 'Delete':
                            Globals.deleteContainer = true
                            break
                        case 'Trivy-Scan':
                            Globals.runTrivyScan = true
                            break
                    }

                    if (Globals.build || Globals.deploy || Globals.runTrivyScan) {
                        echo 'Starting with calulating version'
                        def shortBranchName = env.BRANCH_NAME.replaceAll("[^a-zA-Z0-9]+", "").take(30).toLowerCase()
                        try {
                            Globals.version = sh(script: "git describe --tags --match v[0-9]*", returnStdout: true).trim()
                        } catch (err) {
                            def version = sh(script: "git rev-parse --short HEAD", returnStdout: true).trim()
                            Globals.version = "${shortBranchName}-${version}"
                        }
                        echo "Using version ${Globals.version}"
                        if (env.BRANCH_NAME == 'main') {
                            Globals.imageTag = "${Globals.IMAGE_NAME}:latest"
                        } else {
                            Globals.imageTag = "${Globals.IMAGE_NAME}:${shortBranchName}"
                        }
                        echo "Using container version ${Globals.imageTag}"
                    }
                }
            }
        }

        stage('Build') {
            when { expression { Globals.build } }
            steps {
                echo "Starting with Build image"
                sh """
                podman build --pull --build-arg VERSION=${Globals.version} --target tester -t ${Globals.imageTag}-tester .
                mkdir -p test_reports
                """
                echo "Starting with unit-testing including coverage"
                sh "./run_within_db_pod.sh ${Globals.imageTag}-tester '. ./test_ci.sh && run_tests_with_coverage'"
            }
            post {
                always {
                    junit keepLongStdio: true, testResults: 'test_reports/junit.xml'
                }
            }
        }


        stage('Scan') {
            when { expression { Globals.build } }
            steps {
                script {
                    echo("---- LYNT ----")
                    sh "podman run --rm -v \$(pwd)/test_reports:/src/app-root/test_reports ${Globals.imageTag}-tester sh -c '. ./test_ci.sh && run_pylint'"

                    try {
                        echo("---- TYPING CHECK ----")
                        sh "podman run --rm -v \$(pwd)/test_reports:/src/app-root/test_reports ${Globals.imageTag}-tester sh -c '. ./test_ci.sh && run_mypy'"
                        recordIssues(qualityGates: [[threshold: 10, type: 'TOTAL', unstable: false]], tools: [myPy(pattern: 'test_reports/mypy.log')])
                    }
                    catch (err) {
                        error "Too many mypy issues, exiting now..."
                    }

                    echo("---- SONARQUBE ANALYSIS ----")
                    withSonarQubeEnv("Sonarqube-PROD") {
                        // fix source path in coverage.xml
                        // (required because coverage is calculated using podman which uses a differing file structure)
                        // https://stackoverflow.com/questions/57220171/sonarqube-client-fails-to-parse-pytest-coverage-results
                        sh "sed -i 's/\\/src\\/app-root/.\\//g' test_reports/coverage.xml"
                        sh "${scannerHome}/bin/sonar-scanner"
                    }

                    echo("---- SONARQUBE QUALITY GATE ----")
                    timeout(time: 1, unit: 'HOURS') {
                        // Parameter indicates whether to set pipeline to UNSTABLE if Quality Gate fails
                        // true = set pipeline to UNSTABLE, false = don't
                        waitForQualityGate abortPipeline: Globals.qualityGateAbortPipeline
                    }
                }
            }
        }

        stage('Release') {
            when { expression { Globals.release } }
            steps {
                echo 'Build a wheel and publish'
                script {
                    withCredentials([string(credentialsId: "python-mch-nexus-secret", variable: 'PIP_PWD')]) {
                        runDevScript("build/poetry-lib-release.sh ${env.PIP_USER} $PIP_PWD")
                        Globals.version = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
                        env.TAG_NAME = Globals.version
                    }
                }
            }
        }

        stage('Create Artifacts') {
            when { expression { Globals.build || Globals.deploy} }
            steps {
                script {
                    sh "podman build --pull --target runner --build-arg VERSION=${Globals.version} -t ${Globals.imageTag} ."
                }
            }
        }

        stage('Publish Artifacts') {
            when { expression { Globals.deploy } }
            environment {
                REGISTRY_AUTH_FILE = "$workspace/.containers/auth.json"
            }
            steps {
                script {
                    if (expression { Globals.deploy }) {
                        echo "---- PUBLISH IMAGE ----"
                        withCredentials([usernamePassword(credentialsId: 'openshift-nexus',
                            passwordVariable: 'NXPASS', usernameVariable: 'NXUSER')]) {
                            sh """
                            echo $NXPASS | podman login ${Globals.IMAGE_REPO} -u $NXUSER --password-stdin
                            podman push ${Globals.imageTag}
                            """
                        }
                    }
                }
            }
            post {
                cleanup {
                    sh "podman logout docker-intern-nexus.meteoswiss.ch || true"
                    sh 'oc logout || true'
                }
            }
        }


        stage('Scan Artifacts') {
            when { expression { Globals.runTrivyScan } }
            environment {
                REGISTRY_AUTH_FILE = "$workspace/.containers/auth.json"
                PATH = "$HOME/tools/openshift-client-tools:$HOME/tools/trivy:$PATH"
                HTTP_PROXY = "http://proxy.meteoswiss.ch:8080"
                HTTPS_PROXY = "http://proxy.meteoswiss.ch:8080"
            }
            steps {
                echo "---- TRIVY SCAN ----"
                withCredentials([usernamePassword(credentialsId: 'openshift-nexus',
                    passwordVariable: 'NXPASS', usernameVariable: 'NXUSER')]) {
                    sh "echo $NXPASS | podman login ${Globals.IMAGE_REPO} -u $NXUSER --password-stdin"
                    runDevScript("test/trivyscanner.py ${Globals.imageTag}")
                }
            }
            post {
                cleanup {
                    sh "podman logout docker-intern-nexus.meteoswiss.ch || true"
                }
            }
        }

        stage('Deploy') {
            when { expression { Globals.deploy } }
            steps {
                // TODO RMF-81 add deployment steps
                echo 'Not yet implemented'
            }
        }

        stage('Restart Deployment') {
            when { expression { Globals.restart } }
            steps {
                // TODO RMF-81 add deployment steps
                echo 'Not yet implemented'
            }
        }

        stage('Delete Deployment') {
            when { expression { Globals.deleteContainer } }
            steps {
                // TODO RMF-81 add deployment steps
                echo 'Not yet implemented'
            }
        }
    }


    post {
        cleanup {
            sh "podman image rm -f ${Globals.imageTag}-tester || true"
            sh "podman image rm -f ${Globals.imageTag} || true"
        }
        failure {
            echo 'Sending email'
            sh 'df -h'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
        success {
            echo 'Build succeeded'
        }
    }
}
