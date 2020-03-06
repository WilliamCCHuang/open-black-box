import types


def modify_model(model):
    if 'vgg' in str(model.__class__):
        model = modify_vggs(model)
    
    return model


def modify_vggs(model):
    def features(self, input):
        x = self._features(input)
        return x

    def logits(self, features):
        x = features.view(features.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model