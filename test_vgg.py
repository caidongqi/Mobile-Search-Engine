import logging
import torch
import data
import torchvision
import torchmetrics

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets.vgg import VggDataset
logging.basicConfig(level=logging.INFO, force=True)

device = "cuda:7" if torch.cuda.is_available() else "cpu"

# lora = False
# linear_probing = False

# load_head_post_proc_finetuned = True



# assert not (linear_probing and lora), \
#             "Linear probing is a subset of LoRA training procedure for ImageBind. " \
#             "Cannot set both linear_probing=True and lora=True. "

# if lora and not load_head_post_proc_finetuned:
#     # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
#     lora_factor = 12 / 0.07
# else:
#     # This assumes proper loading of all params but results in shift from original dist in case of LoRA
#     lora_factor = 1
#310种分类
text_list=['warbler chirping', 'people nose blowing', 'electric grinder grinding', 'cattle, bovinae cowbell', 'people hiccup', 'cutting hair with electric trimmers', 'slot machine', 'airplane flyby', 'baby babbling', 'black capped chickadee calling', 'playing piano', 'frog croaking', 'canary calling', 'people babbling', 'orchestra', 'mosquito buzzing', 'francolin calling', 'playing harpsichord', 'bird squawking', 'bouncing on trampoline', 'using sewing machines', 'sailing', 'wood thrush calling', 'mouse pattering', 'pig oinking', 'people slurping', 'door slamming', 'snake rattling', 'playing trumpet', 'playing banjo', 'bathroom ventilation fan running', 'underwater bubbling', 'dog howling', 'playing harmonica', 'missile launch', 'people battle cry', 'magpie calling', 'ocean burbling', 'shot football', 'swimming', 'sea lion barking', 'playing marimba, xylophone', 'disc scratching', 'playing squash', 'dog whimpering', 'playing tabla', 'dog baying', 'people burping', 'people humming', 'turkey gobbling', 'zebra braying', 'cell phone buzzing', 'air horn', 'dog bow-wow', 'skateboarding', 'barn swallow calling', 'playing bongo', 'roller coaster running', 'civil defense siren', 'playing acoustic guitar', 'playing snare drum', 'playing accordion', 'playing vibraphone', 'printer printing', 'car passing by', 'dog barking', 'playing guiro', 'strike lighter', 'metronome', 'arc welding', 'people whispering', 'child singing', 'telephone bell ringing', 'car engine knocking', 'bird chirping, tweeting', 'playing ukulele', 'yodelling', 'blowtorch igniting', 'typing on computer keyboard', 'people sniggering', 'firing muskets', 'ice cracking', 'playing congas', 'train horning', 'people booing', 'thunder', 'playing bass guitar', 'fire truck siren', 'stream burbling', 'driving snowmobile', 'forging swords', 'wind noise', 'scuba diving', 'mynah bird singing', 'playing darts', 'sloshing water', 'cat hissing', 'plastic bottle crushing', 'chinchilla barking', 'subway, metro, underground', 'penguins braying', 'train whistling', 'ambulance siren', 'bull bellowing', 'playing bugle', 'chicken crowing', 'people gargling', 'singing choir', 'train wheels squealing', 'driving buses', 'people running', 'playing zither', 'people cheering', 'cat caterwauling', 'chimpanzee pant-hooting', 'baby crying', 'sliding door', 'playing steel guitar, slide guitar', 'playing didgeridoo', 'people marching', 'police radio chatter', 'playing electronic organ', 'helicopter', 'striking pool', 'playing timpani', 'fireworks banging', 'playing erhu', 'cuckoo bird calling', 'playing lacrosse', 'rowboat, canoe, kayak rowing', 'mouse clicking', 'skiing', 'car engine idling', 'dinosaurs bellowing', 'duck quacking', 'lions roaring', 'mouse squeaking', 'heart sounds, heartbeat', 'tractor digging', 'planing timber', 'horse clip-clop', 'playing djembe', 'squishing water', 'playing theremin', 'otter growling', 'people farting', 'female speech, woman speaking', 'playing steelpan', 'motorboat, speedboat acceleration', 'playing tuning fork', 'playing synthesizer', 'hedge trimmer running', 'ripping paper', 'playing volleyball', 'playing cello', 'people clapping', 'playing bass drum', 'raining', 'toilet flushing', 'reversing beeps', 'typing on typewriter', 'playing badminton', 'smoke detector beeping', 'wind rustling leaves', 'extending ladders', 'writing on blackboard with chalk', 'splashing water', 'elk bugling', 'owl hooting', 'car engine starting', 'vehicle horn, car horn, honking', 'race car, auto racing', 'whale calling', 'footsteps on snow', 'snake hissing', 'playing trombone', 'dog growling', 'playing table tennis', 'people eating noodle', 'spraying water', 'playing clarinet', 'people sobbing', 'sheep bleating', 'electric shaver, electric razor shaving', 'driving motorcycle', 'lawn mowing', 'chopping wood', 'foghorn', 'golf driving', 'fire crackling', 'playing hockey', 'playing drum kit', 'playing hammond organ', 'tap dancing', 'baltimore oriole calling', 'playing saxophone', 'playing tympani', 'crow cawing', 'people coughing', 'playing oboe', 'coyote howling', 'people sneezing', 'cricket chirping', 'running electric fan', 'tornado roaring', 'cat meowing', 'ferret dooking', 'people shuffling', 'wind chime', 'eletric blender running', 'playing shofar', 'cow lowing', 'lions growling', 'parrot talking', 'bird wings flapping', 'sharpen knife', 'singing bowl', 'male singing', 'playing flute', 'goose honking', 'fly, housefly buzzing', 'fox barking', 'hail', 'alligators, crocodiles hissing', 'playing mandolin', 'eating with cutlery', 'ice cream truck, ice cream van', 'air conditioning noise', 'cat purring', 'playing tambourine', 'bee, wasp, etc. buzzing', 'police car (siren)', 'church bell ringing', 'opening or closing drawers', 'baby laughter', 'people screaming', 'playing timbales', 'rope skipping', 'children shouting', 'playing electric guitar', 'playing tennis', 'playing french horn', 'horse neighing', 'airplane', 'tapping guitar', 'sea waves', 'pumping water', 'chopping food', 'playing violin, fiddle', 'striking bowling', 'people eating crisps', 'alarm clock ringing', 'playing washboard', 'cheetah chirrup', 'hammering nails', 'playing bagpipes', 'waterfall burbling', 'railroad car, train wagon', 'people eating apple', 'people whistling', 'playing gong', 'popping popcorn', 'engine accelerating, revving, vroom', 'playing castanets', 'bowling impact', 'beat boxing', 'playing double bass', 'female singing', 'elephant trumpeting', 'volcano explosion', 'woodpecker pecking tree', 'people slapping', 'chicken clucking', 'male speech, man speaking', 'cat growling', 'gibbon howling', 'people crowd', 'firing cannon', 'opening or closing car doors', 'lathe spinning', 'people eating', 'child speech, kid speaking', 'playing harp', 'people giggling', 'lip smacking', 'opening or closing car electric windows', 'playing glockenspiel', 'hair dryer drying', 'cupboard opening or closing', 'playing bassoon', 'cattle mooing', 'people belly laughing', 'playing cymbal', 'chipmunk chirping', 'pheasant crowing', 'eagle screaming', 'people finger snapping', 'skidding', 'playing sitar', 'chainsawing trees', 'playing cornet', 'lighting firecrackers', 'vacuum cleaner cleaning floors', 'goat bleating', 'cap gun shooting', 'pigeon, dove cooing', 'machine gun shooting', 'rapping', 'donkey, ass braying', 'basketball bounce']
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
v_block=len(model.modality_trunks["vision"].blocks)
t_block=len(model.modality_trunks["text"].blocks)
a_block=len(model.modality_trunks["audio"].blocks)
i_block=len(model.modality_trunks["imu"].blocks)

# if lora:
#     model.modality_trunks.update(
#         LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
#                                         layer_idxs={ModalityType.TEXT: [ 1, 2, 3, 4, 5, 6, 7, 8],
#                                                     ModalityType.AUDIO: [1, 2, 3, 4, 5, 6, 7, 8]},
#                                         modality_names=[ModalityType.TEXT, ModalityType.AUDIO]))

#     # Load LoRA params if found
#     LoRA.load_lora_modality_trunks(model.modality_trunks,
#                                    checkpoint_dir="./.checkpoints/lora/imagenet-text15")

#     if load_head_post_proc_finetuned:
#         # Load postprocessors & heads
#         load_module(model.modality_postprocessors, module_name="postprocessors",
#                     checkpoint_dir="./.checkpoints/lora/imagenet-text15")
#         load_module(model.modality_heads, module_name="heads",
#                     checkpoint_dir="./.checkpoints/lora/imagenet-text15")
# elif linear_probing:
#     # Load heads
#     load_module(model.modality_heads, module_name="heads",
#                 checkpoint_dir="./.checkpoints/lora/imagenet-text15")

model.eval()
model.to(device)

'''
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('test.png')
'''

def run_inference(text_class):
    # 
    csv_file_path = "/data/air/pc/Mobile-Search-Engine/datasets/vggsound.csv"
    data_dir="/data/air/pc/Mobile-Search-Engine/datasets/vgg-wav"
    Vgg_dataset = VggDataset(csv_file=csv_file_path,datadir=data_dir,device=device)
    test_dl = DataLoader(dataset=Vgg_dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, persistent_workers=True)

    test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=1000, average="micro").to(device)
    test_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=1000, average="micro").to(device)
    
    test_correct = 0
    test_total = 0
    counts = np.array([])
    counts_5 = np.array([])
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            # if batch_idx > 20:
                # break
            # x = x.to(device)
            target = target.to(device)
            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(x,device),
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            }
            
            embeddings = model(inputs)
            match_value_1 = embeddings[ModalityType.AUDIO]@embeddings[ModalityType.TEXT].T 
            
            # result_1 = match_value_1
            result_1 = torch.softmax(match_value_1, dim=-1)
            _, predicted = torch.max(result_1, -1)
            _, topk_indices = torch.topk(result_1, k=5, dim=-1)
            counts = np.concatenate((counts,predicted.cpu().numpy()==target.cpu().numpy()))
            counts_5 = np.concatenate([counts_5, [any(topk_indices[i] == target[i]) for i in range(topk_indices.size(0))]])

            for i in range(topk_indices.size(0)):
                # Check if any value in the current row of topk_indices is equal to the corresponding value in target
                # if any(topk_indices[i] == target[i]):
                    
                    counts_5 = np.concatenate((counts,any(topk_indices[i] == target[i])))

            #counts_5=np.concatenate((counts,predicted.cpu().numpy()==target.cpu().numpy()))
            acc = test_acc(result_1.argmax(1), target)
            
            recall = test_recall(result_1.argmax(1), target)
            correct = predicted.eq(target).sum()
            test_correct += correct.item()
            test_total += target.size(0)
            logging.info(f"batch_idx = {batch_idx}, test_correct = {test_correct}, test_total = {test_total}, Accuracy = {acc}, Recall = {recall}")
    
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    logging.info(f" Accuracy = {total_acc}, Recall = {total_recall}")
    
    # hist1, bin_edges1 = np.histogram(counts, bins=11)
    # fig1, ax1 = plt.subplots()
    # ax1.hist(bin_edges1[:-1], bin_edges1, weights=hist1, facecolor='skyblue',
    #         alpha=0.7, edgecolor='k')
    # # ax1.set_xlabel('Mass (g)')
    # ax1.set_ylabel('Counts')
    # print(np.mean(counts))
    # plt.savefig('pricisions.pdf')
    
    np.savetxt(f'./results/top1/top1_t{t_block}_a{a_block}.txt',counts,fmt='%d')
    np.savetxt(f'./results/top5/top5_t{t_block}_a{a_block}.txt',counts_5,fmt='%d')
    return test_correct / test_total

def main():
    Accuracy = run_inference(text_list)
    print("Model Performance:", Accuracy)

def print_text_label():
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    datadir = "./.datasets/imagenet"
    test_ds = ImageNet(datadir, split="val", transform=data_transform)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    
    labels = sorted(list(set(batch[1] for batch in test_dl)))
    print(labels)

if __name__ == "__main__":
    main()
    # print_text_label()
