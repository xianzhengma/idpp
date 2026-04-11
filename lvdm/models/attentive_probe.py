import importlib
import logging
import torch
import torch.nn as nn
from src.models.attentive_pooler import AttentivePooler
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AttentiveClassifier(nn.Module):

    def __init__(self, verb_classes: dict, noun_classes: dict, action_classes: dict, embed_dim: int, num_heads: int, depth: int, use_activation_checkpointing: bool):
        super().__init__()
        self.num_verb_classes = len(verb_classes)
        num_noun_classes = len(noun_classes)
        num_action_classes = len(action_classes)
        self.action_only = self.num_verb_classes == 0
        self.pooler = AttentivePooler(num_queries=1 if self.action_only else 3, embed_dim=embed_dim, num_heads=num_heads, depth=depth, use_activation_checkpointing=use_activation_checkpointing)
        if not self.action_only:
            self.verb_classifier = nn.Linear(embed_dim, self.num_verb_classes, bias=True)
            self.noun_classifier = nn.Linear(embed_dim, num_noun_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)

    def forward(self, x):
        if torch.isnan(x).any():
            print('Nan detected at output of encoder')
            exit(1)
        x = self.pooler(x)
        if not self.action_only:
            x_verb, x_noun, x_action = (x[:, 0, :], x[:, 1, :], x[:, 2, :])
            x_verb = self.verb_classifier(x_verb)
            x_noun = self.noun_classifier(x_noun)
            x_action = self.action_classifier(x_action)
            return dict(verb=x_verb, noun=x_noun, action=x_action)
        else:
            x_skip_projection_feature = x[:, 0, :]
            x_action = self.action_classifier(x_skip_projection_feature)
            return dict(action=x_action, skip_projection_feature=x_skip_projection_feature)
