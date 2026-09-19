/* Included by ds4.c after the common GGUF validation helpers. */
static void config_validate_bonsai_model(const ds4_model *m) {
    static const struct { const char *key; uint32_t value; } checks[] = {
        {"qwen35.block_count",64}, {"qwen35.embedding_length",5120},
        {"qwen35.feed_forward_length",17408}, {"qwen35.attention.head_count",24},
        {"qwen35.attention.head_count_kv",4}, {"qwen35.attention.key_length",256},
        {"qwen35.attention.value_length",256}, {"qwen35.rope.dimension_count",64},
        {"qwen35.ssm.conv_kernel",4}, {"qwen35.ssm.state_size",128},
        {"qwen35.ssm.group_count",16}, {"qwen35.ssm.time_step_rank",48},
        {"qwen35.ssm.inner_size",6144}, {"qwen35.full_attention_interval",4},
        {"prism.hadamard.version",1}, {"prism.hadamard.block_size",1024}
    };
    for (size_t i=0;i<sizeof(checks)/sizeof(checks[0]);i++)
        config_expect_u32(checks[i].key,required_u32(m,checks[i].key),checks[i].value);
    static const char *keys[] = {"prism.hadamard.transform","prism.hadamard.axis","prism.hadamard.sign_mode","tokenizer.ggml.pre"};
    static const char *values[] = {"normalized-sylvester-walsh-hadamard","input-last-dimension","explicit","qwen35"};
    for (size_t i=0;i<4;i++) {
        ds4_str v={0};
        if (!model_get_string(m,keys[i],&v) || !ds4_streq(v,values[i])) ds4_die("unsupported Bonsai Hadamard/tokenizer metadata");
    }
    ds4_array_ref sections;
    if (!model_get_array(m,"qwen35.rope.dimension_sections",&sections) ||
        sections.type != GGUF_VALUE_INT32 || sections.len != 4) ds4_die("invalid Bonsai RoPE sections");
    ds4_cursor c=cursor_at(m,sections.data_pos);
    const uint32_t want[4]={11,11,10,0};
    for (int i=0;i<4;i++) { uint32_t n; if (!cursor_u32(&c,&n) || n!=want[i]) ds4_die("unsupported Bonsai RoPE sections"); }
    uint32_t mtp=0;
    if ((model_get_u32(m,"qwen35.nextn_predict_layers",&mtp) && mtp) || model_find_kv(m,"qwen35.attention.recurrent_layers"))
        ds4_die("Bonsai supports the standard 64-layer dense trunk without MTP");
    float eps=0,base=0;
    if (!model_get_f32_compat(m,"qwen35.attention.layer_norm_rms_epsilon",&eps) || eps!=1e-6f ||
        !model_get_f32_compat(m,"qwen35.rope.freq_base",&base) || base!=10000000.0f)
        ds4_die("unsupported Bonsai normalization or RoPE base");
    /* Do not accidentally accept scaled attention or RoPE variants. */
    if (model_find_kv(m,"qwen35.rope.scaling.type") || model_find_kv(m,"qwen35.attention.scale"))
        ds4_die("Bonsai scaled RoPE/attention metadata is not supported");
    g_ds4_shape=(ds4_shape){.name="Ternary Bonsai 2 27B",.family=DS4_MODEL_FAMILY_BONSAI,
        .variant=DS4_VARIANT_BONSAI,.n_layer=64,.n_embd=5120,.n_vocab=248320,
        .n_head=24,.n_head_kv=4,.n_head_dim=256,.n_value_dim=256,.n_rot=64,
        .n_ff_dense=17408,.n_hc=1,.n_lin_k_head=16,.n_lin_v_head=48,
        .n_lin_head_dim=128,.n_lin_conv=4,.n_full_attn_interval=4,
        .rms_eps=eps,.rope_freq_base=base,.rope_orig_ctx=262144};
    memset(g_ds4_compress_ratios,0,sizeof(g_ds4_compress_ratios));
}

static ds4_bonsai_tensor bonsai_bind_tensor(const ds4_model *m, ds4_bonsai_model *b,
        const char *name, uint32_t cols, uint32_t rows, bool matrix, bool folded,
        bool *folds, uint32_t *used) {
    const ds4_tensor *t=required_tensor(m,name);
    if (t->ndim!=(matrix?2u:1u) || t->dim[0]!=cols || (matrix && t->dim[1]!=rows))
        ds4_die("Bonsai tensor dimensions do not match the dense graph");
    if (folded ? (t->type!=142 && t->type!=143) :
        (matrix ? (t->type!=0 && t->type!=30 && t->type!=1) : t->type!=0))
        ds4_die("unsupported Bonsai tensor encoding");
    ds4_bonsai_tensor w={.data=m->map+t->abs_offset,.bytes=t->bytes,.type=t->type,.cols=cols,.rows=rows};
    bool listed=folds[t-m->tensors];
    if (listed!=folded) ds4_die("Bonsai folded-weight list does not match its execution graph");
    if (folded) {
        for (int i=0;i<3;i++) if (b->sign_widths[i]==cols) w.signs=b->signs[i];
        if (!w.signs) ds4_die("missing Bonsai sign vector for a folded weight");
        (*used)++;
    }
    return w;
}

static ds4_bonsai_model *bonsai_bind(const ds4_model *m) {
    ds4_bonsai_model *b=xcalloc(1,sizeof(*b));
    *b=(ds4_bonsai_model){.n_layer=64,.n_embd=5120,.n_vocab=248320,.n_ff=17408,
        .n_head=24,.n_kv_head=4,.head_dim=256,.n_rot=64,.n_k_head=16,.n_v_head=48,
        .ssm_dim=128,.conv_width=4,.full_interval=4,.context=262144,.eps=1e-6f,.rope_base=10000000.0f};
    config_expect_u32("qwen35.context_length",required_u32(m,"qwen35.context_length"),b->context);
    if (!model_get_bool(m,"prism.hadamard.gdn_v_grouped",&b->gdn_v_grouped))
        ds4_die("Bonsai is missing its GDN head-order metadata");
    ds4_array_ref widths,signs,names,inverse;
    if (!model_get_array(m,"prism.hadamard.sign_widths",&widths) || widths.type!=GGUF_VALUE_INT32 || widths.len!=3 ||
        !model_get_array(m,"prism.hadamard.sign_values",&signs) || signs.type!=GGUF_VALUE_INT32 || signs.len!=28672 ||
        !model_get_array(m,"prism.hadamard.weight_names",&names) || names.type!=GGUF_VALUE_STRING || names.len!=401 ||
        !model_get_array(m,"prism.hadamard.inverse_weight_names",&inverse) || inverse.type!=GGUF_VALUE_STRING || inverse.len!=1)
        ds4_die("invalid Bonsai Hadamard tables");
    ds4_cursor wc=cursor_at(m,widths.data_pos),sc=cursor_at(m,signs.data_pos);
    const uint32_t expected[3]={5120,6144,17408};
    for (int i=0;i<3;i++) {
        if (!cursor_u32(&wc,&b->sign_widths[i]) || b->sign_widths[i]!=expected[i]) ds4_die("unsupported Bonsai sign widths");
        b->signs[i]=xmalloc(expected[i]*sizeof(int32_t));
        for (uint32_t j=0;j<expected[i];j++) {
            uint32_t value;
            if (!cursor_u32(&sc,&value) || (value!=1 && value!=UINT32_MAX)) ds4_die("Bonsai signs must be +1 or -1");
            b->signs[i][j]=value==1?1:-1;
        }
    }
    bool *folds=xcalloc((size_t)m->n_tensors,sizeof(bool));
    ds4_cursor nc=cursor_at(m,names.data_pos);
    for (uint64_t i=0;i<names.len;i++) {
        ds4_str name;
        if (!cursor_string(&nc,&name) || name.len>=128) ds4_die("invalid Bonsai folded weight name");
        char key[128]; memcpy(key,name.ptr,(size_t)name.len); key[name.len]=0;
        const ds4_tensor *t=required_tensor(m,key);
        if (folds[t-m->tensors]) ds4_die("duplicate Bonsai folded weight");
        folds[t-m->tensors]=true;
    }
    nc=cursor_at(m,inverse.data_pos);
    ds4_str name;
    if (!cursor_string(&nc,&name) || !ds4_streq(name,"token_embd.weight")) ds4_die("unsupported Bonsai inverse table");
    uint32_t used=0;
#define BIND(field,key,c,r,mat,fold) b->field=bonsai_bind_tensor(m,b,key,c,r,mat,fold,folds,&used)
    BIND(output,"output.weight",5120,248320,true,true);
    BIND(output_norm,"output_norm.weight",5120,1,false,false);
    /* Embedding is transformed after lookup, not before a matmul. */
    const ds4_tensor *emb=required_tensor(m,"token_embd.weight");
    if (folds[emb-m->tensors]) ds4_die("embedding listed as a forward folded weight");
    folds[emb-m->tensors]=true;
    BIND(embedding,"token_embd.weight",5120,248320,true,true);
    for (uint32_t il=0;il<64;il++) {
        char key[128];
#define L(field,suffix,c,r,mat,fold) do { snprintf(key,sizeof(key),"blk.%u.%s",il,suffix); BIND(layer[il].field,key,c,r,mat,fold); } while(0)
        L(norm,"attn_norm.weight",5120,1,false,false);
        L(post_norm,"post_attention_norm.weight",5120,1,false,false);
        L(gate,"ffn_gate.weight",5120,17408,true,true);
        L(up,"ffn_up.weight",5120,17408,true,true);
        L(down,"ffn_down.weight",17408,5120,true,true);
        if ((il+1)%4) {
            L(qkv,"attn_qkv.weight",5120,10240,true,true);
            L(z,"attn_gate.weight",5120,6144,true,true);
            L(out,"ssm_out.weight",6144,5120,true,true);
            L(alpha,"ssm_alpha.weight",5120,48,true,false);
            L(beta,"ssm_beta.weight",5120,48,true,false);
            L(conv,"ssm_conv1d.weight",4,10240,true,false);
            if (b->layer[il].conv.type!=0) ds4_die("Bonsai convolution must be F32");
            L(a,"ssm_a",48,1,false,false); L(dt,"ssm_dt.bias",48,1,false,false);
            L(ssm_norm,"ssm_norm.weight",128,1,false,false);
        } else {
            L(q,"attn_q.weight",5120,12288,true,true);
            L(k,"attn_k.weight",5120,1024,true,true);
            L(v,"attn_v.weight",5120,1024,true,true);
            L(out,"attn_output.weight",6144,5120,true,true);
            L(q_norm,"attn_q_norm.weight",256,1,false,false);
            L(k_norm,"attn_k_norm.weight",256,1,false,false);
        }
#undef L
    }
#undef BIND
    free(folds);
    if (used!=names.len+1) ds4_die("Bonsai folded-weight coverage mismatch");
    return b;
}

static void bonsai_model_free(ds4_bonsai_model *b) {
    if (!b) return;
    for (int i=0;i<3;i++) free(b->signs[i]);
    free(b);
}
