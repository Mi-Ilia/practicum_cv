"""Combo Loss: комбинация Cross-Entropy и Dice Loss

Реализация комбинированного лосса для семантической сегментации:
L = 0.5 * L_CE + 0.5 * L_Dice

Ссылка: https://arxiv.org/html/2312.05391v1
"""

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.losses import CrossEntropyLoss, DiceLoss


@MODELS.register_module()
class ComboLoss(nn.Module):
    """Combo Loss: комбинация Cross-Entropy и Dice Loss
    
    Args:
        ce_weight (float, optional): Вес для Cross-Entropy loss. Defaults to 0.5.
        dice_weight (float, optional): Вес для Dice loss. Defaults to 0.5.
        use_sigmoid (bool, optional): Использовать ли sigmoid для предсказаний. 
            Defaults to False (используется softmax для multi-class).
        class_weight (list[float] | str, optional): Веса классов для CE loss.
            Если строка, загружает веса из файла. Defaults to None.
        reduction (str, optional): Метод редукции для CE loss. 
            Options: "none", "mean", "sum". Defaults to 'mean'.
        loss_weight (float, optional): Общий вес loss. Defaults to 1.0.
        ignore_index (int, optional): Индекс класса для игнорирования. Defaults to 255.
        eps (float, optional): Эпсилон для Dice loss. Defaults to 1e-3.
        loss_name (str, optional): Имя loss для логирования. Defaults to 'loss_combo'.
    """
    
    def __init__(self,
                 ce_weight=0.5,
                 dice_weight=0.5,
                 use_sigmoid=False,
                 class_weight=None,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_combo'):
        super().__init__()
        
        # Проверка, что веса суммируются к 1.0
        assert abs(ce_weight + dice_weight - 1.0) < 1e-6, \
            f"ce_weight ({ce_weight}) + dice_weight ({dice_weight}) должны суммироваться к 1.0"
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name
        
        # Инициализируем CE loss
        self.ce_loss = CrossEntropyLoss(
            use_sigmoid=use_sigmoid,
            use_mask=False,
            reduction=reduction,
            class_weight=class_weight,
            loss_weight=1.0,
            loss_name='loss_ce',
            avg_non_ignore=False
        )
        
        # Инициализируем Dice loss
        self.dice_loss = DiceLoss(
            use_sigmoid=use_sigmoid,
            activate=True,  # Активируем предсказания внутри DiceLoss
            reduction=reduction,
            naive_dice=False,  # Используем стандартный Dice loss (V-Net версию)
            loss_weight=1.0,
            ignore_index=ignore_index,
            eps=eps,
            loss_name='loss_dice'
        )
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Вычисляет комбинированный loss.
        
        Args:
            pred (torch.Tensor): Предсказание модели, shape (N, num_classes, H, W)
            target (torch.Tensor): Ground truth маска, shape (N, H, W)
            weight (torch.Tensor, optional): Веса для каждого пикселя. Defaults to None.
            avg_factor (int, optional): Фактор усреднения. Defaults to None.
            reduction_override (str, optional): Переопределение редукции. Defaults to None.
            ignore_index (int, optional): Переопределение ignore_index. Defaults to None.
            **kwargs: Дополнительные аргументы для CE и Dice losses.
        
        Returns:
            torch.Tensor: Вычисленный комбинированный loss
        """
        # Используем ignore_index из аргументов, если передан, иначе из self.ignore_index
        ignore_idx = ignore_index if ignore_index is not None else self.ignore_index
        
        # Вычисляем CE loss
        ce_loss_val = self.ce_loss(
            pred=pred,
            target=target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_idx,
            **kwargs
        )
        
        # Вычисляем Dice loss
        dice_loss_val = self.dice_loss(
            pred=pred,
            target=target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_idx,
            **kwargs
        )
        
        # Комбинируем losses: L = ce_weight * L_CE + dice_weight * L_Dice
        combo_loss = self.ce_weight * ce_loss_val + self.dice_weight * dice_loss_val
        
        # Применяем общий вес
        return self.loss_weight * combo_loss
    
    @property
    def loss_name(self):
        """Имя loss функции для логирования."""
        return self._loss_name
