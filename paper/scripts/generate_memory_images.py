import sys
sys.path.append('..')
import os
import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS



def run(path:str,dataset_dir:str):
    """ Function to generate memory images for testing images using a given
    model. Memory images show the samples in the memory set that have an
    impact on the current prediction.

    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))    
    # load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)
    
    #saving stuff
    dir_save = "../images/mem_images/"+dataset_name+"/"+modality+"/" + checkpoint['model_name'] + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im
    
    wrong_count = 0
    saved_count = 0

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(test_loader):
            print("Batch:{}/{}".format(batch_idx, len(test_loader)), end='\r')

            if saved_count >= 20:
                print(f"\nSuccessfullly generated {saved_count} images. Exiting...")
                break

            try:
                memory, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(mem_loader)
                memory, _ = next(memory_iter)
            
                    
            images = images.to(device)
            targets = targets.to(device)
            memory = memory.to(device)

            # compute output
            outputs,rw = model(images,memory,return_weights=True)
            _, predictions = torch.max(outputs, 1)

            # compute memory outputs
            mem_val,memory_sorted_index = torch.sort(rw,descending=True)
            
            for ind in range(len(images)):
                true_class_idx = targets[ind].item()
                pred_class_idx = predictions[ind].item()
                absolute_idx = batch_idx * batch_size_test + ind

                if true_class_idx == pred_class_idx:
                    continue
                
                print(f"\n[!] Mistake found! Total mistakes so far: {wrong_count} (Index: {absolute_idx})")
                wrong_count += 1

                if wrong_count <= 20:
                    print(f"    --> Skipping mistake #{wrong_count} per requirements...")
                    continue
                
                input_selected = images[ind].unsqueeze(0)

                try:
                    mem_batch, mem_targets = next(memory_iter)
                except StopIteration:
                    memory_iter = iter(mem_loader)
                    mem_batch, mem_targets = next(memory_iter)
                
                mem_batch = mem_batch.to(device)
                mem_targets = mem_targets.to(device)

                correct_class_mask = (mem_targets == true_class_idx)
                filtered_memory = mem_batch[correct_class_mask]

                if len(filtered_memory) == 0:
                    print(f"    --> [!] No pictures of correct class in this batch. Cannot retest.")
                else:
                    outputs_filtered, _ = model(input_selected, filtered_memory, return_weights=True)
                    _, new_prediction = torch.max(outputs_filtered, 1)

                    if new_prediction.item() == true_class_idx:
                        print(f"    --> EXPERIMENT 1: Success! Filtered memory fixed prediction for index {absolute_idx}")
                    else:
                        print(f"    --> EXPERIMENT 1: Failed. Still guessed {name_classes[new_prediction.item()]}")


                fig = plt.figure(figsize= (2, 4), dpi=300)
                columns = 1
                rows = 2

                print(f"    --> Starting Experiment 2: Brute-forcing random memories...")
                
                max_attempts = 133 #lucky number
                attempts = 0
                found_correction = False
                
                while not found_correction and attempts < max_attempts:
                    attempts += 1
                    
                    try:
                        rand_mem_batch, _ = next(memory_iter)
                    except StopIteration:
                        memory_iter = iter(mem_loader)
                        rand_mem_batch, _ = next(memory_iter)
                        
                    rand_mem_batch = rand_mem_batch.to(device)
                    
                    outputs_rand, rw_rand = model(input_selected, rand_mem_batch, return_weights=True)
                    _, rand_prediction = torch.max(outputs_rand, 1)
                    
                    if rand_prediction.item() == true_class_idx:
                        found_correction = True
                        print(f"    --> EXPERIMENT 2: Success! Found a random batch that fixed it after {attempts} attempts.")
                        mem_val_rand, memory_sorted_index_rand = torch.sort(rw_rand, descending=True)
                        
                        # Get samples with positive impact (Using [0] because script tests 1 image at a time)
                        m_ec_rand = memory_sorted_index_rand[0][mem_val_rand[0] > 0]

                        # Revert normalization and make a grid of the helpful images
                        reduced_mem_rand = undo_normalization(rand_mem_batch[m_ec_rand])
                        npimg_rand = torchvision.utils.make_grid(reduced_mem_rand, nrow=4).cpu().numpy()

                        # Build the image
                        fig_rand = plt.figure(figsize=(2, 4), dpi=300)
                        
                        # Top half: The original test image
                        fig_rand.add_subplot(2, 1, 1)
                        plt.imshow((get_image(input_selected)* 255).astype(np.uint8), interpolation='nearest', aspect='equal')
                        # Title is green to show it was fixed!
                        title_text_rand = f"Idx: {absolute_idx}\nFIXED! True: {name_classes[true_class_idx]}"
                        plt.title(title_text_rand, fontsize=8, color='green', fontweight='bold')
                        plt.axis('off')

                        # Bottom half: The random memory set that saved the day
                        fig_rand.add_subplot(2, 1, 2)
                        plt.imshow((np.transpose(npimg_rand, (1,2,0))* 255).astype(np.uint8), interpolation='nearest', aspect='equal')
                        plt.title('Random Used Samples')
                        plt.axis('off')
                        fig_rand.tight_layout()

                        # Save it with a distinct name so I know it's from Experiment 2
                        file_path_rand = dir_save + f"random_fixed_idx_{absolute_idx}.png"
                        fig_rand.savefig(file_path_rand)
                        plt.close()
                        print(f"    --> Saved the successful random memory batch to {file_path_rand}")

                       


                        
                if not found_correction:
                    print(f"    --> EXPERIMENT 2: Failed. Tried {max_attempts} random batches and none fixed it.")

                

                # M_c u M_e : set of sample with a positive impact on prediction
                m_ec = memory_sorted_index[ind][mem_val[ind]>0]

                # get reduced memory
                reduced_mem = undo_normalization(memory[m_ec])
                npimg = torchvision.utils.make_grid(reduced_mem,nrow=4).cpu().numpy()

                # build and store image

                fig.add_subplot(rows, columns, 1)
                plt.imshow((get_image(input_selected)* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
                
                title_text = f"Idx: {absolute_idx}\nTrue: {name_classes[true_class_idx]}\nPred: {name_classes[pred_class_idx]}"
                plt.title(title_text, fontsize=8, color='red', fontweight='bold')
                plt.axis('off')



                ax2 = fig.add_subplot(rows, columns,2)
                plt.imshow((np.transpose(npimg, (1,2,0))* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
                plt.title('Used Samples')
                plt.axis('off')
                fig.tight_layout()


                file_path = dir_save + f"wrong_idx_{absolute_idx}.png"
                fig.savefig(file_path)
                plt.close()
                
                saved_count += 1
                print(f'\nSaved image {saved_count}/20 (Absolute Index: {absolute_idx})')

                if saved_count >= 20:
                    break



def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)