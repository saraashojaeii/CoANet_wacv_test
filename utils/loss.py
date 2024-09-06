import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='con_ce':
            return self.ConLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        # loss = torch.mean(torch.sum(-target * torch.log(F.softmax(logit, dim=1)), dim=1))
        # loss = torch.mean(torch.sum(-target * nn.LogSoftmax()(logit), dim=1))
        loss = nn.BCEWithLogitsLoss()(logit, target)
        # loss = nn.BCELoss()(logit, target)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())


class BCE_SAC_lcDice(nn.Module):
    def __init__(self, num_classes, alpha, beta, phi, cel, ftl, lcl, K=3):
        super(BCE_SAC_lcDice, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.cel = cel
        self.ftl = ftl
        self.lcl = lcl
        self.K = K

    def lc_dice_loss(self, inputs, targets, alpha=1.0, beta=1.0):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = F.one_hot(targets.squeeze(1).long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Flatten the tensors using reshape
        inputs = inputs.reshape(inputs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        
        # Log-Cosh loss
        log_cosh = torch.log(torch.cosh(inputs - targets))
        log_cosh_loss = torch.mean(log_cosh)
        
        # Dice loss
        intersection = (inputs * targets).sum(dim=1)
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum(dim=1) + targets.sum(dim=1) + 1)
        dice_loss = dice_loss.mean()
        
        # lcDice loss
        lc_dice_loss = alpha * log_cosh_loss + beta * dice_loss
        return lc_dice_loss

    
    def tversky_loss(self, true, logits, alpha, beta, eps=1e-7):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1).cuda()[true.long().squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            device = true.device  
            true = true.squeeze(1).long()
            true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_score = (num / (denom + eps)).mean()
        return (1 - tversky_score)**self.phi

    def weights(self, pred, target, epsilon=1e-6):
        pred_class = torch.argmax(pred, dim=1)
        d = np.ones(self.num_classes)
        for c in range(self.num_classes):
            t = (target == c).sum()
            d[c] = t
        d = d / d.sum()
        d = 1 - d
        return torch.from_numpy(d).float()

    def GapMat(self, pred, target):
        criterion = nn.CrossEntropyLoss(reduction='none')
        target = target.squeeze(1).long()
        L = criterion(pred, target)
        A = torch.argmax(pred, dim=1)
    
        # Ensure the tensor is on the CPU before converting to numpy
        A_cpu = A.cpu().numpy()
        distance_transform = distance_transform_edt(A_cpu == 0)
        threshold = 10
        # Apply the threshold
        distance_transform[distance_transform > threshold] = threshold
        # Invert the distance transform
        distance_transform_inverted = ((threshold - distance_transform) / threshold)
        # Normalize for visualization
        A2 = cv2.normalize(distance_transform_inverted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
        B = np.zeros_like(A_cpu)
        for n in range(A_cpu.shape[0]):
            temp = skeletonize(A_cpu[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)
    
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
    
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        C = torch.where(C == 1, 1, 0).double()
    
        kernel = torch.ones((1, 1, 9, 9), dtype=torch.double).to(pred.device)
    
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=4, dilation=1, groups=1)
        N = N * self.K
    
        temp = torch.where(N == 0, 1, 0)
        W1 = N + temp
    
        W1[W1 == 1] = 0 
        A2 = A2 / 255.0
    
        A2_tensor = torch.tensor(A2, dtype=torch.double).to(pred.device)
        A2_tensor = torch.unsqueeze(A2_tensor, dim=0).unsqueeze(dim=0)
        W1 = W1.squeeze(0).squeeze(0)
        W2 = torch.mul(A2_tensor, W1)
    
        temp2 = torch.where(W2 == 0, 1, 0)
        W = W2 + temp2
    
        output = W * L
        loss = torch.mean(W * L)
        return output


    def forward(self, pred, target):
        # print(self.cel + self.ftl + self.lcl)
        # if (self.cel + self.ftl + self.lcl) != 1:
        #     raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        
        target_squeezed = target.squeeze(1).long()
        loss_seg = nn.CrossEntropyLoss(weight=self.weights(pred, target).cuda())
        ce_seg = loss_seg(pred, target_squeezed)
        
        pred_weighted = self.GapMat(pred, target)
        tv = self.tversky_loss(target, pred_weighted, alpha=self.alpha, beta=self.beta)
        
        # Ensure the target for lc_dice_loss is in the same shape as predictions
        lcd = self.lc_dice_loss(pred, target)
        
        total_loss = (self.cel * ce_seg) + (self.ftl * tv) + (self.lcl * lcd)
        return total_loss


