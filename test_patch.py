import wandb

# 순수하게 방만 파는 코드
wandb.init(project="my_test_project", name="patch-test", save_code=True)

# 아무 의미 없는 로그 하나 찍기
wandb.log({"test": 1})
wandb.finish()