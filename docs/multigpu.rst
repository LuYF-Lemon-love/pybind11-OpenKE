多 GPU 配置
==================================

由于多 GPU 设置依赖于 `accelerate <https://github.com/huggingface/accelerate>`_ ，
因此，您需要首先需要创建并保存一个配置文件：

.. prompt:: bash

	accelerate config

.. _accelerate:

参考配置为：

.. prompt:: bash

    $ accelerate config
    ---------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
    In which compute environment are you running?
    This machine                                                                                                                                       
    ---------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                               
    Which type of machine are you using?
    multi-GPU                                                                                                                                          
    How many different machines will you use (use more than 1 for multi-node training)? [1]:                                                           
    Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes                 
    Do you wish to optimize your script with torch dynamo?[yes/NO]:No                                                                                  
    Do you want to use DeepSpeed? [yes/NO]: No                                                                                                         
    Do you want to use FullyShardedDataParallel? [yes/NO]: No                                                                                          
    Do you want to use Megatron-LM ? [yes/NO]: No                                                                                                      
    How many GPU(s) should be used for distributed training? [1]:2                                                                                     
    What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
    ---------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
    Do you wish to use FP16 or BF16 (mixed precision)?
    no                                                                                                                                                 
    accelerate configuration saved at /home/luyanfeng/.cache/huggingface/accelerate/default_config.yaml                                                
    $
    
然后，您可以开始训练：

.. prompt:: bash

	accelerate launch accelerate_transe_FB15K.py