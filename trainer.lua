require 'neuralconvo'
require 'xlua'
require 'optim'

-- load the JSON library.
local Json = require("json")

local JsonStorage = {}

-- Function to save a table.&nbsp; Since game settings need to be saved from session to session, we will
-- use the Documents Directory
JsonStorage.saveTable = function(t, filename)
    -- local path = system.pathForFile( filename, system.DocumentsDirectory)
    local file = io.open(filename, "w")

    if file then
        local contents = Json.encode(t)
        file:write( contents )
        io.close( file )
        return true
    else
        return false
    end
end

function train()
    local progress = {}
    progress['progress'] = 0
    JsonStorage.saveTable(progress, "progress.json")
    options = {}
    options.dataset = 100
    options.hiddenSize = 50
    options.maxEpoch = 50

    options.maxVocabSize = 0
    options.cuda = false
    options.opencl = false
    options.learningRate = 0.001
    options.gradientClipping = 5
    options.momentum = 0.9
    options.minLR = 0.00001
    options.saturateEpoch = 20
    options.batchSize = 10
    options.gpu = 0


    if options.dataset == 0 then
        options.dataset = nil
    end

    -- Data
    print("-- Loading dataset")
    dataset = neuralconvo.DataSet(neuralconvo.CornellMovieDialogs("data/cornell_movie_dialogs"),
        {
            loadFirst = options.dataset,
            maxVocabSize = options.maxVocabSize
        })

    print("\nDataset stats:")
    print("  Vocabulary size: " .. dataset.wordsCount)
    print("         Examples: " .. dataset.examplesCount)

    -- Model
    model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize)
    model.goToken = dataset.goToken
    model.eosToken = dataset.eosToken

    -- Training parameters
    if options.batchSize > 1 then
        model.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
    else
        model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    end

    local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
    local minMeanError = nil

    -- Enabled CUDA
    if options.cuda then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(options.gpu + 1)
        model:cuda()
    elseif options.opencl then
        require 'cltorch'
        require 'clnn'
        cltorch.setDevice(options.gpu + 1)
        model:cl()
    end

    -- Run the experiment
    local optimState = {learningRate=options.learningRate,momentum=options.momentum}
    for epoch = 1, options.maxEpoch do
        collectgarbage()

        local nextBatch = dataset:batches(options.batchSize)
        local params, gradParams = model:getParameters()

        -- Define optimizer
        local function feval(x)
            if x ~= params then
                params:copy(x)
            end

            gradParams:zero()
            local encoderInputs, decoderInputs, decoderTargets = nextBatch()

            if options.cuda then
                encoderInputs = encoderInputs:cuda()
                decoderInputs = decoderInputs:cuda()
                decoderTargets = decoderTargets:cuda()
            elseif options.opencl then
                encoderInputs = encoderInputs:cl()
                decoderInputs = decoderInputs:cl()
                decoderTargets = decoderTargets:cl()
            end

            -- Forward pass
            local encoderOutput = model.encoder:forward(encoderInputs)
            model:forwardConnect(encoderInputs:size(1))
            local decoderOutput = model.decoder:forward(decoderInputs)
            local loss = model.criterion:forward(decoderOutput, decoderTargets)

            local avgSeqLen = nil
            if #decoderInputs:size() == 1 then
                avgSeqLen = decoderInputs:size(1)
            else
                avgSeqLen = torch.sum(torch.sign(decoderInputs)) / decoderInputs:size(2)
            end
            loss = loss / avgSeqLen

            -- Backward pass
            local dloss_doutput = model.criterion:backward(decoderOutput, decoderTargets)
            model.decoder:backward(decoderInputs, dloss_doutput)
            model:backwardConnect(encoderInputs:size(1))
            model.encoder:backward(encoderInputs, encoderOutput:zero())

            gradParams:clamp(-options.gradientClipping, options.gradientClipping)

            return loss,gradParams
        end

        -- run epoch

        print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch ..
                "  (LR= " .. optimState.learningRate .. ")")
        print("")

        local errors = {}
        local timer = torch.Timer()

        for i=1, dataset.examplesCount/options.batchSize do
            collectgarbage()
            local _,tloss = optim.adam(feval, params, optimState)
            err = tloss[1] -- optim returns a list

            model.decoder:forget()
            model.encoder:forget()

            table.insert(errors,err)
            xlua.progress(i * options.batchSize, dataset.examplesCount)
        end

        xlua.progress(dataset.examplesCount, dataset.examplesCount)
        timer:stop()

        errors = torch.Tensor(errors)
        print("\n\nFinished in " .. xlua.formatTime(timer:time().real) ..
                " " .. (dataset.examplesCount / timer:time().real) .. ' examples/sec.')
        print("\nEpoch stats:")
        print("  Errors: min= " .. errors:min())
        print("          max= " .. errors:max())
        print("       median= " .. errors:median()[1])
        print("         mean= " .. errors:mean())
        print("          std= " .. errors:std())
        print("          ppl= " .. torch.exp(errors:mean()))

        -- Save the model if it improved.
        if minMeanError == nil or errors:mean() < minMeanError then
            print("\n(Saving model ...)")
            params, gradParams = nil,nil
            collectgarbage()
            -- Model is saved as CPU
            model:double()
            torch.save("data/model.t7", model)
            collectgarbage()
            if options.cuda then
                model:cuda()
            elseif options.opencl then
                model:cl()
            end
            collectgarbage()
            minMeanError = errors:mean()
        end

        optimState.learningRate = optimState.learningRate + decayFactor
        optimState.learningRate = math.max(options.minLR, optimState.learningRate)

        -- save to file

        progress['progress'] = epoch / (options.maxEpoch + 1)
        JsonStorage.saveTable(progress, "progress.json")
    end
end

return train