TextureCube g_SkyTexture : register(t0);
SamplerState g_Sampler : register(s0);

struct PixelShaderInput
{
    float3 TexCoord : TEXCOORD;
};

float4 main(PixelShaderInput IN) : SV_Target
{
    return g_SkyTexture.Sample(g_Sampler, IN.TexCoord);
}