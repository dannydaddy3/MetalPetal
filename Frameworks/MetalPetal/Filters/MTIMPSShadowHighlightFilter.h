//
//  MTIMPSShadowHighlightFilter.h
//  MetalPetal
//
//  Created by Yu Ao on 2018/8/23.
//

#import <Foundation/Foundation.h>
#import <MetalPetal/MTIFilter.h>

NS_ASSUME_NONNULL_BEGIN

@interface MTIMPSShadowHighlightFilter : NSObject <MTIUnaryFilter>

@property (nonatomic) float radius;

@property (nonatomic) float shadow;

@property (nonatomic) float highlight;

@property (nonatomic) float midtoneContrast;

@property (nonatomic) float colorCorrection;

@end

NS_ASSUME_NONNULL_END
