//
//  MTIMPSShadowHighlightFilter.m
//  MetalPetal
//
//  Created by Yu Ao on 2018/8/23.
//

#import "MTIMPSShadowHighlightFilter.h"
#import <MetalPetal/MTIMPSGaussianBlurFilter.h>
#import <MetalPetal/MTIRenderPipelineKernel.h>
#import <MetalPetal/MTIFunctionDescriptor.h>
#import <MetalPetal/MTIImage.h>

@interface MTIMPSShadowHighlightFilter ()

@property (nonatomic, strong) MTIMPSGaussianBlurFilter *blurFilter;

@end

@implementation MTIMPSShadowHighlightFilter
@synthesize inputImage = _inputImage;
@synthesize outputPixelFormat = _outputPixelFormat;

+ (MTIRenderPipelineKernel *)kernel {
    return [[MTIRenderPipelineKernel alloc] initWithVertexFunctionDescriptor:[[MTIFunctionDescriptor alloc] initWithName:MTIFilterPassthroughVertexFunctionName]
                                                  fragmentFunctionDescriptor:[[MTIFunctionDescriptor alloc] initWithName:@"shadowHighlightAdjust"]];
}

- (instancetype)init {
    if (self = [super init]) {
        _radius = 30;
        _blurFilter = [[MTIMPSGaussianBlurFilter alloc] init];
        _blurFilter.radius = _radius;
    }
    return self;
}

- (MTIImage *)outputImage {
    if (!self.inputImage) {
        return nil;
    }
    self.blurFilter.radius = self.radius;
    self.blurFilter.inputImage = self.inputImage;
    MTIImage *blurredImage = self.blurFilter.outputImage;
    return [MTIMPSShadowHighlightFilter.kernel applyToInputImages:@[self.inputImage, blurredImage]
                                                       parameters:@{@"shadow": @(self.shadow),
                                                                    @"highlight": @(self.highlight)}
                                          outputTextureDimensions:self.inputImage.dimensions
                                                outputPixelFormat:self.outputPixelFormat];
}

@end
