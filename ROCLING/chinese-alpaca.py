import os
os.environ['TRANSFORMERS_CACHE'] = '/nfs/Workspace/wesley'
# os.environ['HF_DATASETS_CACHE']= '/nfs/Workspace/wesley'
# os.environ['HF_HOME']= '/nfs/Workspace/wesley'
# os.environ['PYTORCH_KERNEL_CACHE_PATH']= '/nfs/Workspace/wesley'

# import locale
# locale.getpreferredencoding = lambda: "UTF-8"

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained('sharpbai/chinese-alpaca-plus-lora-7b-merged')
device = torch.device(0)
base_model = LlamaForCausalLM.from_pretrained('sharpbai/chinese-alpaca-plus-lora-7b-merged',
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map='auto')
base_model = base_model.eval()
generation_config = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    do_sample = False,
    num_beams=5,
    repetition_penalty=1.5,
    max_new_tokens=400)
# input_text = "告诉我中国和美国分别各有哪些优点缺点"
prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)
def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


# sentenes = [
#    "今天天氣很適合出去玩",
#    "台灣撐個屁 天塌下來有高個子擋",
#    "幫我買飲料",
#    "前陣子還有人瘦到剩皮包骨 一定是假新聞",
#    "他腿好粗喔...對了原PO有點矮",
#    "黑道8＋9一堆瘦皮猴，會有力氣拉弓",
#    "而且也不帥 是個又醜又肥的死阿宅",
#    "美國就邪惡國家 只是大腿太粗人人抱",
#    "不然拍低能兒的精子來騙喜憨兒算不算創意 ? 快槍決肥豬啊",
#    "鳥仔腳就只能永遠穿長襪之類的才好看",
#    "黃猴子瘦不拉機只能靠太極拳打手槍",
#    "空空流太中二了讓給你，我比較喜歡烏龍妹",
#    "換個例子試試看 胖是異常 醜是異常 窮是異常 ...略",]




import pandas as pd
df = pd.read_csv("./llama_ver2_Q1.csv",index_col=[0],low_memory=False)
# df = df[0:10]

for sentene, label, comprehensive, predicted, response_Q1 in zip(df.sentene, df.label, df.comprehensive, df.predicted, df.response):
    if predicted == 0:
        fat_skinny = 0
        response = "Q1 判斷無歧視"
    elif predicted == 3:
        fat_skinny = 3
        response = "Q1 無法判斷"
    elif predicted == 1:
        input_text = generate_prompt(instruction=f"""
                                     身體歧視可分為歧視瘦子，跟歧視胖子兩種類型，且兩種類型不會相互重複
                                     下列文字是一篇確定有身體歧視的文章，判斷該文章是哪種類型，只回答歧視瘦子或歧視胖子兩者其中之一，不要有多餘解釋和其他類別
                                     {sentene}""")
        # input_ids = input_ids.to('cuda')
        inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
        generation_output = base_model.generate(
            input_ids = inputs["input_ids"].to('cuda'),
            attention_mask = inputs['attention_mask'],
            **generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        response = output.split("### Response:")[1].strip()
        if "歧視胖" in response:
            fat_skinny = 1
        elif "歧視瘦" in response:
            fat_skinny = 2
        else:
            fat_skinny = 3
    # print(label,type(label))
    # if label == 0 or label == 4  or label == '0' or label == "4": 
    #     label = 0

    dir_keys = ["sentene", "label", "fat_skinny", "response", "predicted", "comprehensive", "response_Q1"]
    dir_values =[sentene, label, fat_skinny, response, predicted, comprehensive, response_Q1]
    dic = dict(zip(dir_keys, dir_values))
    df_re = pd.DataFrame(dic,index=[0])
    df_re.to_csv("llama_ver2_Q2_original.csv",mode='a', encoding='utf-8')
    print(f"""句子:\n{sentene}\n{response}\nlabel:{label}\nfat_skinny:{fat_skinny}\n\n\n""")










# import pandas as pd
# df = pd.read_csv("./llms_df.csv",index_col=[0],low_memory=False)


# for sentene,label in zip(df.content, df.label):
#     input_text = generate_prompt(instruction=
#                                 f"""
#                                 身體羞辱定義：[是一種非重複性行為在這種行為中加害者不請自來的對目標身體，表達負面意見或評論。加害者不一定有意傷害被害者，但被害者認為該評論是負面、具有冒犯性或使其產生身體羞恥感。因此身體羞辱的範圍從善意的建議到惡意的羞辱]
#                                 下列文字是一篇文章，根據身體羞辱定義判斷該文章是否涉及有關於身體羞辱的歧視?回答涉及歧視或未涉及歧視
#                                 {sentene} 
#                                 """)
#     # input_ids = input_ids.to('cuda')
#     inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
#     generation_output = base_model.generate(
#         input_ids = inputs["input_ids"].to('cuda'),
#         attention_mask = inputs['attention_mask'],
#         **generation_config
#     )
#     s = generation_output[0]
#     output = tokenizer.decode(s,skip_special_tokens=True)
#     response = output.split("### Response:")[1].strip()

#     if "未涉及歧視" in response or "沒有關於" in response :
#         predicted = 0
#     elif "這篇文章涉及身體羞辱的歧視" in  response:
#         predicted = 1
#     else:
#         print(sentene,'\n',response,"\n\n\n")
#         predicted = 3

#     dir_keys = ["sentene", "label", "predicted", "response"]
#     dir_values =[sentene, label, predicted, response]
#     dic = dict(zip(dir_keys, dir_values))
#     df_re = pd.DataFrame(dic,index=[0])
#     df_re.to_csv("llama_ver2_Q1_original.csv",mode='a', encoding='utf-8')

#     print(f"""句子:\n{sentene}\n{response}\npredicted:{predicted}\nlabel:{label}\n""")