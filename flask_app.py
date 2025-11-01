runtime: python39
instance_class: F2
entrypoint: gunicorn -b :$PORT main:app

env_variables:
  TELEGRAM_TOKEN: "8396377413:AAGtSWquXrolQR2LlqRdh3a75zd8Zt5UOfg"

automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 1
  max_instances: 3

handlers:
- url: /.*
  script: auto