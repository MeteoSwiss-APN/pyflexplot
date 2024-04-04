module "mch-ecr-repository" {
  source  = "app.terraform.io/meteoswiss/mch-solution-ecr/aws"
  version = "0.0.3"

  app_name                         = "flexpart-cosmo-pyflexplot"
  environment                      = var.environment
  image_tag_mutability             = "MUTABLE"
  create_lifecycle_policy_untagged = true
  create_lifecycle_policy_tagged   = false
  image_lifecycle_days             = 30
  encryption_type                  = "AES256"
  kms_key                          = null
}
