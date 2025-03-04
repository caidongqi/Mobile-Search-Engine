import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import csv
import torch
import torch.nn as nn
import torchaudio
class VggDataset(Dataset):
    def __init__(self, csv_file,device='cpu',datadir=''):
        # 初始化操作，可以在这里加载数据集
        self.dir=datadir
        dataset = os.listdir(datadir)
        self.datadir=sorted(dataset)
        self.data = pd.read_csv(csv_file,sep=',') # 假设数据集以CSV文件形式提供
        self.text_list=['warbler chirping', 'people nose blowing', 'electric grinder grinding', 'cattle, bovinae cowbell', 'people hiccup', 'cutting hair with electric trimmers', 'slot machine', 'airplane flyby', 'baby babbling', 'black capped chickadee calling', 'playing piano', 'frog croaking', 'canary calling', 'people babbling', 'orchestra', 'mosquito buzzing', 'francolin calling', 'playing harpsichord', 'bird squawking', 'bouncing on trampoline', 'using sewing machines', 'sailing', 'wood thrush calling', 'mouse pattering', 'pig oinking', 'people slurping', 'door slamming', 'snake rattling', 'playing trumpet', 'playing banjo', 'bathroom ventilation fan running', 'underwater bubbling', 'dog howling', 'playing harmonica', 'missile launch', 'people battle cry', 'magpie calling', 'ocean burbling', 'shot football', 'swimming', 'sea lion barking', 'playing marimba, xylophone', 'disc scratching', 'playing squash', 'dog whimpering', 'playing tabla', 'dog baying', 'people burping', 'people humming', 'turkey gobbling', 'zebra braying', 'cell phone buzzing', 'air horn', 'dog bow-wow', 'skateboarding', 'barn swallow calling', 'playing bongo', 'roller coaster running', 'civil defense siren', 'playing acoustic guitar', 'playing snare drum', 'playing accordion', 'playing vibraphone', 'printer printing', 'car passing by', 'dog barking', 'playing guiro', 'strike lighter', 'metronome', 'arc welding', 'people whispering', 'child singing', 'telephone bell ringing', 'car engine knocking', 'bird chirping, tweeting', 'playing ukulele', 'yodelling', 'blowtorch igniting', 'typing on computer keyboard', 'people sniggering', 'firing muskets', 'ice cracking', 'playing congas', 'train horning', 'people booing', 'thunder', 'playing bass guitar', 'fire truck siren', 'stream burbling', 'driving snowmobile', 'forging swords', 'wind noise', 'scuba diving', 'mynah bird singing', 'playing darts', 'sloshing water', 'cat hissing', 'plastic bottle crushing', 'chinchilla barking', 'subway, metro, underground', 'penguins braying', 'train whistling', 'ambulance siren', 'bull bellowing', 'playing bugle', 'chicken crowing', 'people gargling', 'singing choir', 'train wheels squealing', 'driving buses', 'people running', 'playing zither', 'people cheering', 'cat caterwauling', 'chimpanzee pant-hooting', 'baby crying', 'sliding door', 'playing steel guitar, slide guitar', 'playing didgeridoo', 'people marching', 'police radio chatter', 'playing electronic organ', 'helicopter', 'striking pool', 'playing timpani', 'fireworks banging', 'playing erhu', 'cuckoo bird calling', 'playing lacrosse', 'rowboat, canoe, kayak rowing', 'mouse clicking', 'skiing', 'car engine idling', 'dinosaurs bellowing', 'duck quacking', 'lions roaring', 'mouse squeaking', 'heart sounds, heartbeat', 'tractor digging', 'planing timber', 'horse clip-clop', 'playing djembe', 'squishing water', 'playing theremin', 'otter growling', 'people farting', 'female speech, woman speaking', 'playing steelpan', 'motorboat, speedboat acceleration', 'playing tuning fork', 'playing synthesizer', 'hedge trimmer running', 'ripping paper', 'playing volleyball', 'playing cello', 'people clapping', 'playing bass drum', 'raining', 'toilet flushing', 'reversing beeps', 'typing on typewriter', 'playing badminton', 'smoke detector beeping', 'wind rustling leaves', 'extending ladders', 'writing on blackboard with chalk', 'splashing water', 'elk bugling', 'owl hooting', 'car engine starting', 'vehicle horn, car horn, honking', 'race car, auto racing', 'whale calling', 'footsteps on snow', 'snake hissing', 'playing trombone', 'dog growling', 'playing table tennis', 'people eating noodle', 'spraying water', 'playing clarinet', 'people sobbing', 'sheep bleating', 'electric shaver, electric razor shaving', 'driving motorcycle', 'lawn mowing', 'chopping wood', 'foghorn', 'golf driving', 'fire crackling', 'playing hockey', 'playing drum kit', 'playing hammond organ', 'tap dancing', 'baltimore oriole calling', 'playing saxophone', 'playing tympani', 'crow cawing', 'people coughing', 'playing oboe', 'coyote howling', 'people sneezing', 'cricket chirping', 'running electric fan', 'tornado roaring', 'cat meowing', 'ferret dooking', 'people shuffling', 'wind chime', 'eletric blender running', 'playing shofar', 'cow lowing', 'lions growling', 'parrot talking', 'bird wings flapping', 'sharpen knife', 'singing bowl', 'male singing', 'playing flute', 'goose honking', 'fly, housefly buzzing', 'fox barking', 'hail', 'alligators, crocodiles hissing', 'playing mandolin', 'eating with cutlery', 'ice cream truck, ice cream van', 'air conditioning noise', 'cat purring', 'playing tambourine', 'bee, wasp, etc. buzzing', 'police car (siren)', 'church bell ringing', 'opening or closing drawers', 'baby laughter', 'people screaming', 'playing timbales', 'rope skipping', 'children shouting', 'playing electric guitar', 'playing tennis', 'playing french horn', 'horse neighing', 'airplane', 'tapping guitar', 'sea waves', 'pumping water', 'chopping food', 'playing violin, fiddle', 'striking bowling', 'people eating crisps', 'alarm clock ringing', 'playing washboard', 'cheetah chirrup', 'hammering nails', 'playing bagpipes', 'waterfall burbling', 'railroad car, train wagon', 'people eating apple', 'people whistling', 'playing gong', 'popping popcorn', 'engine accelerating, revving, vroom', 'playing castanets', 'bowling impact', 'beat boxing', 'playing double bass', 'female singing', 'elephant trumpeting', 'volcano explosion', 'woodpecker pecking tree', 'people slapping', 'chicken clucking', 'male speech, man speaking', 'cat growling', 'gibbon howling', 'people crowd', 'firing cannon', 'opening or closing car doors', 'lathe spinning', 'people eating', 'child speech, kid speaking', 'playing harp', 'people giggling', 'lip smacking', 'opening or closing car electric windows', 'playing glockenspiel', 'hair dryer drying', 'cupboard opening or closing', 'playing bassoon', 'cattle mooing', 'people belly laughing', 'playing cymbal', 'chipmunk chirping', 'pheasant crowing', 'eagle screaming', 'people finger snapping', 'skidding', 'playing sitar', 'chainsawing trees', 'playing cornet', 'lighting firecrackers', 'vacuum cleaner cleaning floors', 'goat bleating', 'cap gun shooting', 'pigeon, dove cooing', 'machine gun shooting', 'rapping', 'donkey, ass braying', 'basketball bounce']
        self.device=device

    def __len__(self):
        # 返回数据集的长度
        file_list = os.listdir(self.dir)
        
        # 使用列表推导式过滤出所有文件，而不包括子文件夹  
        return len(file_list)
    

    def __getitem__(self, idx):
        # 根据索引获取单个样本
        # sample = self.data.iloc[idx]
        # audio=self.data
        #dir_path=self.datadir[idx]
        dir=self.datadir[idx]
        dir_path = os.path.join(self.dir, self.datadir[idx])
        for row_number, row in self.data.iterrows():
            # 检查文件名是否在当前行的第一列
            names_id=dir[:11]
            names_num=dir[12:18]
            if row['name'].startswith(names_id):
                # 返回找到的行数
                if  int(row['num'])==int(names_num):
                      label = row['class']
                      index = self.text_list.index(label)
                      return dir_path,index
        
        
        # 在这里可以进行数据转换操作，如果定义了 transform

        

# # 使用示例
# csv_file_path = "/data/air/pc/Mobile-Search-Engine/datasets/vggsound.csv"
# data_dir="/data/air/pc/Mobile-Search-Engine/datasets/vgg-wav"
# device="cuda:5"
# Vgg_dataset = VggDataset(csv_file=csv_file_path,datadir=data_dir,device=device)
# test_dl = DataLoader(dataset=Vgg_dataset, batch_size=64, shuffle=False, drop_last=False,
#         num_workers=4, pin_memory=True, persistent_workers=True)

# with torch.no_grad():
#         for batch_idx, (x, target) in enumerate(test_dl):
#             x=x.to(device)
#             target=[t.to(device) for t in target]