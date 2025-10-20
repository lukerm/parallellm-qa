prod: sync-templates-prod deploy-stack-prod

sync-templates-prod:
	aws s3 sync ./templates s3://pllm-cfn-template-bucket-euwest1-prod/stacks/qa-monitoring --profile pllm-prod

deploy-stack-prod:
	if aws cloudformation describe-stacks --stack-name parallellm-qa-monitoring-stack-prod --profile pllm-prod >/dev/null 2>&1; then \
		echo "Stack exists, updating..."; \
		aws cloudformation update-stack \
		--stack-name parallellm-qa-monitoring-stack-prod \
		--template-body file://./stack.yaml \
		--parameters file://./params.prod.json \
		--capabilities CAPABILITY_NAMED_IAM \
		--profile pllm-prod; \
	else \
		echo "Stack does not exist, creating..."; \
		aws cloudformation create-stack \
		--stack-name parallellm-qa-monitoring-stack-prod \
		--template-body file://./stack.yaml \
		--parameters file://./params.prod.json \
		--capabilities CAPABILITY_NAMED_IAM \
		--profile pllm-prod; \
	fi

