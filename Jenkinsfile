"""
This Jenkinsfile builds an image containing pyflexplot, runs the tests in the container, and can deploy the image to AWS.
Deployments to target environments other than AWS are not yet implemented in this Jenkinsfile.
"""

class Globals {
    // constants
    static final String PROJECT = 'pyflexplot'
    static final String IMAGE_REPO = 'docker-intern-nexus.meteoswiss.ch'
    static final String IMAGE_NAME = 'docker-intern-nexus.meteoswiss.ch/flexpart-cosmo/pyflexplot'

    static final String AWS_IMAGE_NAME = 'mch-meteoswiss-flexpart-cosmo-pyflexplot-repository'
    static final String AWS_REGION = 'eu-central-2'

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

    // the AWS container registry image tag
    static String awsEcrImageTag = ''

    // the AWS ECR repository name
    static String awsEcrRepo = ''

    // the target environment to deploy (e.g., devt, depl, prod)
    static String deployEnv = ''
}

@Library('dev_tools@main') _
pipeline {
    agent { label 'podman' }

    options {
        // New jobs should wait until older jobs are finished
        disableConcurrentBuilds()
        // Discard old builds
        buildDiscarder(logRotator(artifactDaysToKeepStr: '7', artifactNumToKeepStr: '1', daysToKeepStr: '45', numToKeepStr: '10'))
        // Timeout the pipeline build after 1 hour
        timeout(time: 1, unit: 'HOURS')
    }

    stages {

        stage('Deploy') {
            environment {
                HTTPS_PROXY="http://proxy.meteoswiss.ch:8080"
                AWS_DEFAULT_OUTPUT="json"
                AWS_CA_BUNDLE="/etc/ssl/certs/MCHRoot.crt"
                PATH = "/opt/maker/tools/terraform:/opt/maker/tools/aws:$PATH"
            }
            steps {
                withVault(
                    configuration: [vaultUrl: 'https://vault.apps.cp.meteoswiss.ch',
                                    vaultCredentialId: 'flexpart-cosmo-approle',
                                    engineVersion: 2],
                    vaultSecrets: [
                        [
                            path: "flexpart-cosmo/${params.environment}-secrets", engineVersion: 2, secretValues: [
                                [envVar: 'TF_TOKEN_app_terraform_io', vaultKey: 'terraform-token'],
                                [envVar: 'TF_WORKSPACE', vaultKey: 'terraform-workspace-pyflexplot'],
                                [envVar: 'AWS_ACCESS_KEY_ID', vaultKey: 'jenkins-aws-access-key'],
                                [envVar: 'AWS_SECRET_ACCESS_KEY', vaultKey: 'jenkins-aws-secret-key']
                            ]
                        ]
                    ]
                ) {
                    sh """
                        echo \$AWS_ACCESS_KEY_ID >> out
                        echo \$AWS_SECRET_ACCESS_KEY >> out
                        echo \$TF_TOKEN_app_terraform_io >> out

                        cat out
                    """
                    error("Testing security")
                }
            }
        }

    }


    post {
        failure {
            echo 'Sending email'
            sh 'df -h'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
    }
}
