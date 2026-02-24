cbuffer SceneCB : register(b0)
{
    matrix ModelMatrix;
    matrix ModelViewMatrix;
    matrix InverseTransposeModelMatrix;
    matrix ModelViewProjectionMatrix;
    matrix ViewMatrix;
    matrix ProjectionMatrix;
};

struct SkyboxVertex
{
    float3 Position : POSITION;
};

struct VertexShaderOutput
{
    float3 TexCoord : TEXCOORD;
    float4 Position : SV_POSITION;
};

VertexShaderOutput main(SkyboxVertex IN)
{
    VertexShaderOutput OUT;

    OUT.TexCoord = IN.Position;

    matrix viewNoTranslation = ViewMatrix;
    // /!\ HLSL matrices are column-major by default
    viewNoTranslation[0][3] = 0;
    viewNoTranslation[1][3] = 0;
    viewNoTranslation[2][3] = 0;

    float4 viewPos = mul(viewNoTranslation, float4(IN.Position, 1.0));
    float4 clipPos = mul(ProjectionMatrix, viewPos);

    OUT.Position = clipPos.xyww; // force depth to 1.0

    return OUT;
}