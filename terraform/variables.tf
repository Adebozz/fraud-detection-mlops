variable "region" {
  description = "AWS region (London)"
  type        = string
  default     = "eu-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "fraud-detection"
}

variable "kubernetes_version" {
  description = "EKS control-plane version"
  type        = string
  default     = "1.31"
}

variable "node_instance_type" {
  description = "Worker node instance type (t3.medium = cheapest sensible choice)"
  type        = string
  default     = "t3.medium"
}

variable "node_count" {
  description = "Desired worker nodes"
  type        = number
  default     = 2
}
