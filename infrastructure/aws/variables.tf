variable "environment" {
  type        = string
  default     = "devt"
  description = "Environment that resources will be deployed to"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "The image tag to be used"
}

locals {
  service_name     = "dispersionmodelling-flexpart-cosmo-pyflexplot"
  service_name_tag = "dispersionmodelling-flexpart-cosmo-pyflexplot"
  resource_suffix  = "-${var.environment}"
}
