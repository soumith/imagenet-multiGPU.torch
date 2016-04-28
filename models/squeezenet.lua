-- implementation of squeezenet proposed in: http://arxiv.org/abs/1602.07360

local function fire(ch, s1, e1, e3)
        local net = nn.Sequential()
        net:add(nn.SpatialConvolution(ch, s1, 1, 1))
        net:add(nn.ReLU(true))
        local exp = nn.Concat(2)
        exp:add(nn.SpatialConvolution(s1, e1, 1, 1))
        exp:add(nn.SpatialConvolution(s1, e3, 3, 3, 1, 1, 1, 1))
        net:add(exp)
        net:add(nn.ReLU(true))
        return net
end


local function bypass(net)
        local cat = nn.ConcatTable()
        cat:add(net)
        cat:add(nn.Identity())
        local seq = nn.Sequential()
        seq:add(cat)
        seq:add(nn.CAddTable(true))
        return seq
end

local function squeezenet(output_classes)
        local net = nn.Sequential()
        net:add(nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 0, 0)) -- conv1
        net:add(nn.ReLU(true))
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(fire(96, 16, 64, 64))  --fire2
        net:add(bypass(fire(128, 16, 64, 64)))  --fire3
        net:add(fire(128, 32, 128, 128))  --fire4
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(bypass(fire(256, 32, 128, 128)))  --fire5
        net:add(fire(256, 48, 192, 192))  --fire6
        net:add(bypass(fire(384, 48, 192, 192)))  --fire7
        net:add(fire(384, 64, 256, 256))  --fire8
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(bypass(fire(512, 64, 256, 256)))  --fire9
        net:add(nn.Dropout())
        net:add(nn.SpatialConvolution(512, output_classes, 1, 1, 1, 1, 1, 1)) --conv10
        net:add(nn.ReLU(true))
        net:add(nn.SpatialAveragePooling(14, 14, 1, 1))
        net:add(nn.View(output_classes))
        net:add(nn.LogSoftMax())
        return net
end

function createModel(nGPU)
    local model = squeezenet(nClasses)
    model:cuda()
    model = makeDataParallel(model, nGPU)
    model.imageSize = 256
    model.imageCrop = 224
    return model
end
