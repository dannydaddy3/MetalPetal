//
//  Shaders.metal
//  MetalPetal
//
//

#include <metal_stdlib>
#include "MTIShaderLib.h"

using namespace metal;

namespace metalpetal {

    namespace shadowhighlight {
        
        static half4 convertFromRGBToYIQ(half4 src) {
            half3 pix2;
            half4 pix = src;
            pix.xyz = sqrt(fmax(pix.xyz, 0.000000e+00f));
            pix2 = ((pix.x* half3(2.990000e-01f, 5.960000e-01f, 2.120000e-01f))+ (pix.y* half3(5.870000e-01f, -2.755000e-01f, -5.230000e-01f)))+ (pix.z* half3(1.140000e-01f, -3.210000e-01f, 3.110000e-01f));
            return half4(pix2, pix.w);
        }
        
        static half4 convertFromYIQToRGB(half4 src) {
            half4 color, pix;
            pix = src;
            color.xyz = ((pix.x* half3(1.000480e+00f, 9.998640e-01f, 9.994460e-01f))+ (pix.y* half3(9.555580e-01f, -2.715450e-01f, -1.108030e+00f)))+ (pix.z* half3(6.195490e-01f, -6.467860e-01f, 1.705420e+00f));
            color.xyz = fmax(color.xyz, half3(0.000000e+00f));
            color.xyz = color.xyz* color.xyz;
            color.w = pix.w;
            return color;
        }
        
        fragment half4 shadowHighlightAdjust(VertexOut vertexIn [[stage_in]],
                                             texture2d<half, access::sample> sourceTexture [[texture(0)]],
                                             texture2d<half, access::sample> blurTexture [[texture(1)]],
                                             sampler sourceSampler [[sampler(0)]],
                                             sampler blurSampler [[sampler(1)]],
                                             constant float &shadow [[buffer(0)]],
                                             constant float &highlight [[buffer(1)]]) {
            half4 source = sourceTexture.sample(sourceSampler, vertexIn.textureCoordinate);
            half4 blur = blurTexture.sample(blurSampler, vertexIn.textureCoordinate);
            half4 sourceYIQ = convertFromRGBToYIQ(source);
            half4 blurYIQ = convertFromRGBToYIQ(blur);
            half highlights_sign_negated = copysign(1.0, -highlight);
            half shadows_sign = copysign(1.0f, shadow);
            //constexpr half whitepoint = 1.0;
            constexpr half compress = 0.5;
            constexpr half low_approximation = 0.01f;
            constexpr half shadowColor = 1.0;
            constexpr half highlightColor = 1.0;
            half tb0 = 1.0 - blurYIQ.x;
            if (tb0 < 1.0 - compress) {
                half highlights2 = highlight * highlight;
                half highlights_xform = min(1.0f - tb0 / (1.0f - compress), 1.0f);
                while (highlights2 > 0.0f) {
                    half lref, href;
                    half chunk, optrans;
                    
                    float la = sourceYIQ.x;
                    float la_abs;
                    float la_inverted = 1.0f - la;
                    float la_inverted_abs;
                    half lb = (tb0 - 0.5f) * highlights_sign_negated * sign(la_inverted) + 0.5f;
                    
                    la_abs = abs(la);
                    lref = copysign(la_abs > low_approximation ? 1.0f / la_abs : 1.0f / low_approximation, la);
                    
                    la_inverted_abs = abs(la_inverted);
                    href = copysign(la_inverted_abs > low_approximation ? 1.0f / la_inverted_abs : 1.0f / low_approximation, la_inverted);
                    
                    chunk = highlights2 > 1.0f ? 1.0f : highlights2;
                    optrans = chunk * highlights_xform;
                    highlights2 -= 1.0f;
                    
                    sourceYIQ.x = la * (1.0 - optrans) + (la > 0.5f ? 1.0f - (1.0f - 2.0f * (la - 0.5f)) * (1.0f - lb) : 2.0f * la * lb) * optrans;
                    
                    sourceYIQ.y = sourceYIQ.y * (1.0f - optrans)
                    + sourceYIQ.y * (sourceYIQ.x * lref * (1.0f - highlightColor)
                               + (1.0f - sourceYIQ.x) * href * highlightColor) * optrans;
                    
                    sourceYIQ.z = sourceYIQ.z * (1.0f - optrans)
                    + sourceYIQ.z * (sourceYIQ.x * lref * (1.0f - highlightColor)
                               + (1.0f - sourceYIQ.x) * href * highlightColor) * optrans;
                }
            }
            if (tb0 > compress) {
                half shadows2 = shadow * shadow;
                half shadows_xform = min(tb0 / (1.0f - compress) - compress / (1.0f - compress), 1.0f);
                
                while (shadows2 > 0.0f) {
                    half lref, href;
                    half chunk, optrans;
                    
                    float la = sourceYIQ.x;
                    float la_abs;
                    float la_inverted = 1.0f - la;
                    float la_inverted_abs;
                    half lb = (tb0 - 0.5f) * shadows_sign * sign(la_inverted) + 0.5f;
                    
                    la_abs = abs(la);
                    lref = copysign(la_abs > low_approximation ? 1.0f / la_abs : 1.0f / low_approximation, la);
                    
                    la_inverted_abs = abs(la_inverted);
                    href = copysign(la_inverted_abs > low_approximation ? 1.0f / la_inverted_abs : 1.0f / low_approximation,
                                     la_inverted);
                    
                    chunk = shadows2 > 1.0f ? 1.0f : shadows2;
                    optrans = chunk * shadows_xform;
                    shadows2 -= 1.0f;
                    
                    sourceYIQ.x = la * (1.0 - optrans)
                    + (la > 0.5f ? 1.0f - (1.0f - 2.0f * (la - 0.5f)) * (1.0f - lb) : 2.0f * la * lb) * optrans;
                    
                    sourceYIQ.y = sourceYIQ.y * (1.0f - optrans)
                    + sourceYIQ.y * (sourceYIQ.x * lref * shadowColor
                               + (1.0f - sourceYIQ.x) * href * (1.0f - shadowColor)) * optrans;
                    
                    sourceYIQ.z = sourceYIQ.z * (1.0f - optrans)
                    + sourceYIQ.z * (sourceYIQ.x * lref * shadowColor
                               + (1.0f - sourceYIQ.x) * href * (1.0f - shadowColor)) * optrans;
                }
            }
            return convertFromYIQToRGB(sourceYIQ);
        }
    }
}
