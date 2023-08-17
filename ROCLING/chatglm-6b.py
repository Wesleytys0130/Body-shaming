import os
os.environ['TRANSFORMERS_CACHE'] = '/nfs/Workspace/wesley'
os.environ['HF_DATASETS_CACHE']= '/nfs/Workspace/wesley'
os.environ['HF_HOME']= '/nfs/Workspace/wesley'
os.environ['PYTORCH_KERNEL_CACHE_PATH']= '/nfs/Workspace/wesley'
# /nfs/Workspace/wesley


# sentenes = [
#    "台灣撐個屁 天塌下來有高個子擋",
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

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


import pandas as pd
df = pd.read_csv("./chatglm_ver2.csv",index_col=[0],low_memory=False)
print(len(df.label))
# df = df[0:10]

for sentene, label, comprehensive, predicted, response_Q1 in zip(df.sentene, df.label, df.comprehensive, df.predicted, df.response):
   if predicted == 0:
      fat_skinny = 0
      response = "Q1 判斷無歧視"
   elif predicted == 3:
      fat_skinny = 3
      response = "Q1 無法判斷"
   elif predicted == 1:
      response, history = model.chat(tokenizer,f"""
                                    身體羞辱可分為歧視瘦子，跟歧視胖子兩種類型，且兩種類型不會相互重複
                                    下列文字是一篇有身體羞辱歧視的文章，判斷該文章是哪種類型，只回答其中之一
                                    {sentene}""",
                                    history=[],
                                    # top_p=0.3,
                                    temperature=0.3)
      if "歧視胖" in response:
         fat_skinny = 1
      elif "歧視瘦" in response:
         fat_skinny = 2
      else:
         fat_skinny = 3
   # print(label,type(label))
   # if label == 0 or label == 4  or label == '0' or label == "4": 
   #    label = 0

   dir_keys = ["sentene", "label", "fat_skinny", "response", "predicted", "comprehensive", "response_Q1"]
   dir_values =[sentene, label, fat_skinny, response, predicted, comprehensive, response_Q1]
   dic = dict(zip(dir_keys, dir_values))
   df_re = pd.DataFrame(dic,index=[0])
   df_re.to_csv("chatglm_2.csv",mode='a', encoding='utf-8')
   print(f"""句子:\n{sentene}\n{response}\nlabel:{label}\nfat_skinny:{fat_skinny}\n\n\n""")



# import pandas as pd
# df = pd.read_csv("./llms_df.csv",index_col=[0],low_memory=False)
# # df = df[0:10]

# for sentene,label in zip(df.content, df.label):
#    response, history = model.chat(tokenizer,
#                                  f"""
#                                  身體羞辱定義：[是一種非重複性行為在這種行為中加害者不請自來的對目標身體，表達負面意見或評論。加害者不一定有意傷害被害者，但被害者認為該評論是負面、具有冒犯性或使其產生身體羞恥感。因此身體羞辱的範圍從善意的建議到惡意的羞辱]
#                                  下列文字是一篇文章，根據身體羞辱定義判斷該文章是否涉及有關於身體羞辱的歧視?回答涉及歧視或未涉及歧視
#                                  {sentene} 
#                                  """, 
#                                  history=[],
#                                  # top_p=0.3,
#                                  temperature=0.3)
#    #沒有涉及|沒有關於|不符合|不涉及|未涉及
#    #涉及
#    if "沒有涉及" in response or "沒有關於" in response or "不符合" in response or "不涉及" in response or "未涉及" in response or "没有提及" in response or "沒有提及" in response or "无法" in response :
#       predicted = 0
#    elif "涉及" in  response or "提到" in  response or "表明" in  response:
#       predicted = 1
#    else:
#       print(sentene,'\n',response,"\n\n\n")
#       predicted = 3

#    dir_keys = ["sentene", "label", "predicted", "response"]
#    dir_values =[sentene, label, predicted, response]
#    dic = dict(zip(dir_keys, dir_values))
#    df_re = pd.DataFrame(dic,index=[0])
#    df_re.to_csv("chatglm_ver2.csv",mode='a', encoding='utf-8')
#    print(f"""句子:\n{sentene}\n{response}\npredicted:{predicted}\nlabel:{label}\n""")