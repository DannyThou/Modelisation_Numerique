def get_output_width(input_width, kernel_size, stride=1, padding=0, dilation=1):
    """
    - input_wdith: taille de l'entrée
    - kernel_size: taille du noyau
    - stride: pas effectué par le noyau convolution
    - padding: nombre de pixels de padding (0 pour valid, varie pour same)
    - dilation: Espacement entre les pixels du noyau
    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    **Attention: stride=kernel_size pour max pooling par défaut**
    """

    return int(
        (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )