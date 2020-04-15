import abc


class BaseFeatureExtractor(abc.ABC):

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, x):
        return self.forward(x)

    @abc.abstractclassmethod
    def features(self, x):
        pass

    @abc.abstractclassmethod
    def classifier(self, features):
        pass

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)

        return output


class VGGFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, model):
        super(VGGFeatureExtractor, self).__init__(model)

    def features(self, x):
        return self.model.features(x)

    def classifier(self, features):
        x = self.model.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)

        return x