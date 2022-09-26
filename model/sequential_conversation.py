import pytorch_lightning as pl

from model.conversation import ConversationModel


class SequentialConversationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conversation_model = ConversationModel()

    def forward(self):
        return self.conversation_model()
