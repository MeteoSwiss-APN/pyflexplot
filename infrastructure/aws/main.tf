# TODO RMF-81 Move this into IaC separate project? -> then the IaC should also push the image (which needs to be pulled from nexus first)
terraform {
  required_version = "~> 1.0"

  cloud {
    organization = "meteoswiss"

    workspaces {
      name = "flexpart-cosmo-pyflexplot-devt"
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-central-2"
}
