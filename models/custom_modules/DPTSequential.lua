require 'nn'

local DPTSequential, parent = torch.class('nn.DPTSequential', 'nn.Sequential')

function DPTSequential:updateGradInput(input, gradOutput)
   self:backward(input, gradOutput)
   return self.gradInput
end

function DPTSequential:accGradParameters(input, gradOutput, scale)
   -- a no-op
end
