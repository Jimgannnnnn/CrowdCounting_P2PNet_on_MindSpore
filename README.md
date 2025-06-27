这里转换的是一个P2PNet模型，需要有以下模型mindspore版本的内容：
1、mindspore版本的代码
2、由于这个模型权重分成了两部分，一个是P2PNet的权重，一个是基座模型vgg16bn的权重，这里需要有现有的vgg16bn的mindspore权重。如果没有多个权重文件就可以忽略

基本的权重迁移逻辑就是：
两者的模型权重数据是一致的，只有权重索引（keys）的命名方式不同，所以我们只需要把不同索引下的数据一一对应地迁移即可

文件结构：
P2PNet_torch    #包含了模型原torch权重和代码，这里我就不上传了，因为转换部分只调用了模型加载的接口
   |--P2PNet_ms    #文件夹，包含模型mindspore版本的代码
   |--p2p_pth_weight_out.py   #导出pytorch版本模型权重的索引（keys）
   |--p2p_ckpt_weight_out.py  #导出mindspore版本模型权重的索引（keys）
   |--p2p_keys_map.py    #索引的映射表。导出模型索引后，根据一定规则进行索引的一一映射（这个可以自己总结一下映射规律）
   |--p2p_to_ckpt.py     #转换过程的主要代码

更多细节可以去看代码里的注释
