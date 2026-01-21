idx = 0
deploy_map = {}
model_names = ['llama-8b', 'vit-image', 'point-lidar', 'radar-former']
for m in model_names:
    for p in [1, 2, 3]:
        deploy_map[idx] = (m, p)
        idx += 1
print(deploy_map)