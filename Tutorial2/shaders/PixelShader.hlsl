cbuffer SceneCB : register(b0)
{
    matrix ModelMatrix;
    matrix ModelViewMatrix;
    matrix InverseTransposeModelMatrix;
    matrix ModelViewProjectionMatrix;

    float3 CameraPosition;
    float _pad1; // to align to 16 bytes
};

cbuffer DirectLight : register(b1)
{
    float3 LightDirection; // light direction
    float _pad2; // to align to 16 bytes
    float3 LightColor; // light color
    float LightIntensity; // light intensity
};

cbuffer Material : register(b2)
{
    float3 BaseColor;
    float Roughness;
    float Metallic;
    float3 _pad0; // to align to 16 bytes
};

static const float PI = 3.14159265359;

// GGX / Trowbridge-Reitz normal distribution function (NDF) (https://learnopengl.com/PBR/Theory)
// Approximates the relative surface area of microfacets exactly aliggned to the halfway vector h.
float ThrowbridgeReitzNDF(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (https://learnopengl.com/PBR/Theory)
// Approximates the relativve surface area whete its micro surface-details overshadow each other.
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f; // for direct lighting
    // float k = (roughness * roughness) / 2.0f; // for IBL
    return NdotV / (NdotV * (1.0f - k) + k);
}

// Smith's method for geometry function (https://learnopengl.com/PBR/Theory)
// Takes into account both the view direction (geometry obstruction) and the light direction (geometry shadowing).
float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Fresnel-Schlick's approximation ((https://learnopengl.com/PBR/Theory)
// Describes the ratio of light that gets reflected over the light that gets refracted.
// F0 is the reflection coefficient at normal incidence, it is based on the IOR (indices of refraction) of the material.
float3 FresnelSchlick(float VdotH, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - VdotH, 5.0f);
}

float3 BRDFDirect(float3 N, float3 V, float3 L)
{
    float3 H = normalize(V + L);
    
    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));
    
    float3 F0 = { 0.04, 0.04, 0.04 }; // default value for dielectrics
    F0 = lerp(F0, BaseColor, Metallic);
    
    float D = ThrowbridgeReitzNDF(NdotH, Roughness);
    float G = GeometrySmith(NdotV, NdotL, Roughness);
    float F = FresnelSchlick(VdotH, F0);
    
    // Cook-Torrance microfacet specular : specular reflectance from a surface modeled as a collection of microfacets.
    float3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-5);
    
    // Lambertian diffuse
    float3 kd = (1.0f - F) * (1.0f - Metallic);
    float3 diff = kd * BaseColor / PI;
    
    return (diff + spec) * NdotL;
}

struct PixelShaderInput
{
    float4 PositionVS : POSITION;
    float4 NormalVS : NORMAL;
    float4 WorldPosVS : WORLD_POSITION;
};

float4 main( PixelShaderInput IN ) : SV_Target
{
    float3 color = 0;
        
    float3 N = normalize(IN.NormalVS);
    float3 V = normalize(CameraPosition - IN.WorldPosVS.xyz);
    float3 L = normalize(-LightDirection);
    
    color += BRDFDirect(N, V, L);
        
    float3 radiance = LightColor * LightIntensity;
        
    color *= radiance;
        
    return float4(color, 1.0f);
}