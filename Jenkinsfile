class Globals {
    // Pin mchbuild to stable version to avoid breaking changes
    static String mchbuildPipPackage = 'mchbuild>=0.12.2,<0.13.0'
//    static String mchbuildPipPackage = 'mchbuild==0.1.dev510+g9d58a9272'

    // sets to abort the pipeline if the Sonarqube QualityGate fails
    static boolean qualityGateAbortPipeline = false

    // the default python version
    static String pythonVersion = '3.10'

    // Name of the container image
    static String containerImageName= ''

    // Semantic version of the artifact
    static String semanticVersion = ''

    static String PIP_INDEX_URL = 'https://hub.meteoswiss.ch/nexus/repository/python-all/simple'
}

String rebuild_cron = env.BRANCH_NAME == "main" ? "@midnight" : ""

@Library('dev_tools@main') _
pipeline {
    agent { label 'podman' }

    triggers { cron(rebuild_cron) }

    options {
        // New jobs should wait until older jobs are finished
        disableConcurrentBuilds()
        // Discard old builds
        buildDiscarder(logRotator(artifactDaysToKeepStr: '7', artifactNumToKeepStr: '1', daysToKeepStr: '45', numToKeepStr: '10'))
        // Timeout the pipeline build after 1 hour
        timeout(time: 1, unit: 'HOURS')
        gitLabConnection('CollabGitLab')
    }

    environment {
        PATH = "$workspace/.venv-mchbuild/bin:$PATH"
        HTTP_PROXY = 'http://proxy.meteoswiss.ch:8080'
        HTTPS_PROXY = 'http://proxy.meteoswiss.ch:8080'
        NO_PROXY = '.meteoswiss.ch,localhost'
        SCANNER_HOME = tool name: 'Sonarqube-certs-PROD', type: 'hudson.plugins.sonar.SonarRunnerInstallation'
    }

    stages {
        stage('Preflight') {
            steps {
                updateGitlabCommitStatus name: 'Build', state: 'running'

                script {
                    echo '---- INSTALL MCHBUILD ----'
                    sh """
                    python -m venv .venv-mchbuild
                    PIP_INDEX_URL=${Globals.PIP_INDEX_URL} \
                      .venv-mchbuild/bin/pip install --upgrade "${Globals.mchbuildPipPackage}"
                    """

                    echo '---- INITIALIZE PARAMETERS ----'

                    if (env.TAG_NAME) {
                        echo "Detected release build triggered from tag ${env.TAG_NAME}."
                        def isMajorMinorPatch = sh(
                            script: "mchbuild -s version=${env.TAG_NAME} -g isMajorMinorPatch build.checkGivenSemanticVersion",
                            returnStdout: true
                        )
                        if (isMajorMinorPatch != 'true') {
                            currentBuild.result = 'ABORTED'
                            error('Build aborted because release builds are only triggered for tags of the form <major>.<minor>.<patch>.')
                        }
                        Globals.semanticVersion  = env.TAG_NAME
                    } else {
                        echo "Detected development build triggered from branch."
                        Globals.semanticVersion = sh(
                            script: 'mchbuild -g semanticVersion build.getSemanticVersion',
                            returnStdout: true
                        )
                    }

                    Globals.containerImageName = sh(
                        script: 'mchbuild -g containerImageName build.getImageName',
                        returnStdout: true
                    )
                    echo "Using semantic version: ${Globals.semanticVersion}"
                    echo "Using container image name: ${Globals.containerImageName}"
                }
            }
        }

        stage('Build') {
            steps {
                echo '---- BUILDING CONTAINER IMAGES ----'
                sh """
                    mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} build.artifacts
                """

                echo("---- RUNNING UNIT TESTS & COLLECTING COVERAGE ----")
                sh """
                    mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} test.unit
                """
            }
            post {
                always {
                    junit keepLongStdio: true, testResults: 'test_reports/junit*.xml'
                }
            }
        }

        stage('Scan') {
            steps {
                echo '---- LINT & TYPE CHECK ----'
                sh "mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} test.lint"
                script {
                    try {
                        recordIssues(qualityGates: [[threshold: 10, type: 'TOTAL', unstable: false]], tools: [myPy(pattern: 'test_reports/mypy.log')])
                    }
                    catch (err) {
                        error "Too many mypy issues, exiting now..."
                    }
                }

                echo("---- SONARQUBE ANALYSIS ----")
                withSonarQubeEnv("Sonarqube-PROD") {
                    // fix source path in coverage.xml
                    // (required because coverage is calculated using podman which uses a differing file structure)
                    // https://stackoverflow.com/questions/57220171/sonarqube-client-fails-to-parse-pytest-coverage-results
                    sh "sed -i 's/\\/src\\/app-root/.\\//g' test_reports/coverage.xml"
                    sh "${SCANNER_HOME}/bin/sonar-scanner"
                }

                echo("---- SONARQUBE QUALITY GATE ----")
                timeout(time: 1, unit: 'HOURS') {
                    // Parameter indicates whether to set pipeline to UNSTABLE if Quality Gate fails
                    // true = set pipeline to UNSTABLE, false = don't
                    waitForQualityGate abortPipeline: Globals.qualityGateAbortPipeline
                }
            }
        }

        stage('Publish Artifacts') {
            environment {
                REGISTRY_AUTH_FILE = "$workspace/.containers/auth.json"
            }
            steps {
                echo "---- PUBLISHING CONTAINER IMAGES ----"
                withCredentials([usernamePassword(credentialsId: 'openshift-nexus',
                                                  passwordVariable: 'NXPASS',
                                                  usernameVariable: 'NXUSER')]) {
                    sh """
                        mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} publish.artifacts
                    """
                }
            }
        }

        stage('Release') {
            when {
                // This will only execute the stage if TAG_NAME is present
                expression { return env.TAG_NAME != null }
            }
            steps {
                script {
                    echo("---- PUBLISH SBOM ----")
                    withCredentials([string(
                            credentialsId: 'dependency-track-token-prod',
                            variable: 'DTRACK_TOKEN')]) {
                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                sh "mchbuild verify.publishSbom -s version=${Globals.semanticVersion}"
                        }
                    }
                }
            }
        }
    }

    post {
        cleanup {
            sh """
            mchbuild -s semanticVersion=${Globals.semanticVersion} -s containerImageName=${Globals.containerImageName} clean
            """
            cleanWs()
        }
        aborted {
            updateGitlabCommitStatus name: 'Build', state: 'canceled'
        }
        failure {
            updateGitlabCommitStatus name: 'Build', state: 'failed'
            echo 'Sending email'
            sh 'df -h'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                to: env.BRANCH_NAME == 'main' ?
                    sh(script: "mchbuild -g notifyOnNightlyFailure", returnStdout: true) : '',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
        success {
            echo 'Build succeeded'
            updateGitlabCommitStatus name: 'Build', state: 'success'
        }
    }
}
