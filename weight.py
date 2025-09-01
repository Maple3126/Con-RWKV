import torch

# 載入檔案
path = '/home/RWKV-ASR/output/rwkv-adapter-200.pth'
state_dict = torch.load(path, map_location='cpu')  # 可改成 cuda if needed

# 列出所有參數名稱
# for name in state_dict:
#      print(name)
print(state_dict['language_model.blocks.0.depthwise_conv.weight'].view(-1)[:5])
print(state_dict['language_model.blocks.23.depthwise_conv.weight'].view(-1)[:5])
print(state_dict['language_model.blocks.23.depthwise_conv.weight'])
print("conv")

print(state_dict['language_model.blocks.22.att.time_state'].view(-1)[:5])
print("llm")

print(state_dict['speech_encoder.adapter.0.weight'].view(-1)[:5])
print("adapter")
